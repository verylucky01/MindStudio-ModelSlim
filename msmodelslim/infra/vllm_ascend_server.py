#  Copyright (c) 2025-2025 Huawei Technologies Co., Ltd.
#  #
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#  #
#  http://www.apache.org/licenses/LICENSE-2.0
#  #
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

import json
import time
from pathlib import Path
from typing import Literal, Dict, Annotated

import requests
from pydantic import BaseModel, Field, AfterValidator

from msmodelslim.app.auto_tuning.evaluation_service_infra import EvaluateContext
from msmodelslim.utils.exception import ConfigError
from msmodelslim.utils.logging import logger_setter, get_logger
from msmodelslim.utils.plugin import TypedConfig
from msmodelslim.utils.security import AsyncProcess, build_safe_url, safe_get
from msmodelslim.utils.validation.pydantic import (
    is_safe_host,
    is_safe_endpoint,
    is_port,
    greater_than_zero,
)


class VllmAscendConfig(BaseModel):
    type: TypedConfig.TypeField = Literal['vllm-ascend']
    entrypoint: str = "vllm.entrypoints.openai.api_server"
    env_vars: Dict = Field(default_factory=dict)
    served_model_name: str = 'served_model_name'
    host: Annotated[str, AfterValidator(is_safe_host)] = "localhost"
    port: Annotated[int, AfterValidator(is_port)] = 1234
    health_check_endpoint: Annotated[
        str, AfterValidator(is_safe_endpoint)] = "/v1/models"  # vLLM OpenAI-compatible health check
    startup_timeout: Annotated[
        int,
        AfterValidator(greater_than_zero),
    ] = 600
    args: Dict = Field(default_factory=dict)


# 健康检查轮询间隔（秒）
HEALTH_CHECK_INTERVAL = 5
# HTTP 请求超时时间（秒）
HTTP_REQUEST_TIMEOUT = 3


@logger_setter()
class VllmAscendServer:
    """
    配置驱动的 VLLM-Ascend 服务器启动器。

    职责:
    1. 从配置中构建环境变量 (env_vars)。
    2. 从配置中构建命令行参数 (args)，处理 bool, str, dict。
    3. 启动服务并等待其就绪 (health check)。
    4. 停止服务。
    """

    def __init__(self,
                 context: EvaluateContext,
                 server_config: VllmAscendConfig,
                 model_path: Path,
                 log_file_path: Path
                 ):
        """
        Args:
            context: 评估上下文
            server_config: VLLM 服务器配置
            model_path: 量化后模型的路径
            log_file_path: 本次运行的 vllm 日志文件路径
        """
        if not model_path.exists():
            raise ConfigError(f"Model path does not exist: {model_path}")

        self.config = server_config
        self.model_path = model_path
        self.log_file = log_file_path

        # 安全地构造健康检查 URL
        self.health_check_url = self._build_health_check_url()
        self.startup_timeout = self.config.startup_timeout

        # 构建命令选项和环境变量
        cmd_options = self._build_command_options()

        # 初始化异步进程管理器（使用安全接口）
        self.process = AsyncProcess(
            binary=f"python",
            log_file=str(self.log_file),
            options=cmd_options,
            env=self.config.env_vars or None
        )
        get_logger().debug(f"VLLM command options: {cmd_options}")

    def start(self):
        """启动 VLLM 进程并等待其就绪。"""
        get_logger().info(f"Starting VLLM server for model: {self.model_path}")
        self.process.start()

        get_logger().info(f"Waiting for server to be ready at {self.health_check_url} ...")
        if not self._wait_for_ready():
            get_logger().error(f"VLLM server failed to start. Check log: {self.log_file}")
            # 尝试停止僵尸进程
            try:
                self.stop()
            except Exception as e:
                get_logger().warning(f"Failed to stop process during cleanup: {e}")
            return False

        get_logger().info("VLLM server started successfully.")
        return True

    def stop(self):
        """停止 VLLM 进程。"""
        get_logger().info("Stopping VLLM server...")
        self.process.stop()
        get_logger().info("VLLM server stopped.")

    def _build_health_check_url(self) -> str:
        """
        安全地构建健康检查 URL，防止 URL 注入攻击。
        使用安全模块的 URL 构建函数。

        Returns:
            健康检查 URL
        """
        return build_safe_url(
            host=self.config.host,
            port=self.config.port,
            endpoint=self.config.health_check_endpoint,
            scheme='http'
        )

    def _build_command_options(self) -> dict:
        """
        构建命令选项字典，用于安全命令执行。

        Returns:
            选项字典，格式为 {option_name: value}
        """
        options = {
            "-m": self.config.entrypoint,
            "--model": str(self.model_path),
            "--port": str(self.config.port)
        }

        # 遍历配置中的 'args' 来构建其他参数
        for key, value in self.config.args.items():
            if value is True:
                # e.g., trust-remote-code: true -> --trust-remote-code
                options[f"--{key}"] = None
            elif value is False or value is None:
                # e.g., enable-prefix-caching: false -> (被忽略)
                continue
            elif isinstance(value, dict):
                # e.g., additional_config: {...} -> --additional_config='{"...":...}'
                # 序列化为紧凑的 JSON（AsyncProcess 会进行安全验证）
                json_str = json.dumps(value, separators=(',', ':'))
                options[f"--{key}"] = json_str
            else:
                # e.g., tp: 2 -> --tp 2
                options[f"--{key}"] = str(value)

        return options

    def _wait_for_ready(self) -> bool:
        """
        轮询健康检查接口，等待服务器就绪。
        使用安全请求模块防止 SSRF 和其他网络攻击。
        
        Returns:
            True 如果服务器成功启动，False 如果超时
        """
        start_time = time.time()
        while time.time() - start_time < self.startup_timeout:
            try:
                # 使用安全请求函数，自动应用安全配置
                resp = safe_get(
                    self.health_check_url,
                    timeout=HTTP_REQUEST_TIMEOUT
                )
                if resp.status_code == 200:
                    return True
                get_logger().debug(
                    f"Health check returned status {resp.status_code}, "
                    f"retrying..."
                )
            except requests.ConnectionError:
                # 服务器尚未监听端口，继续等待
                pass
            except requests.Timeout:
                get_logger().debug("Health check request timed out, retrying...")
            except requests.TooManyRedirects:
                # 重定向过多，可能是攻击尝试
                get_logger().warning(
                    f"Too many redirects for health check URL: {self.health_check_url}"
                )
            except requests.SSLError as e:
                # SSL 错误（如果使用 HTTPS）
                get_logger().warning(f"SSL verification failed: {e}")
            except requests.RequestException as e:
                # 其他请求错误（如 DNS 解析失败等）
                get_logger().debug(f"Health check request error: {e}")

            time.sleep(HEALTH_CHECK_INTERVAL)

        get_logger().error(
            f"Server startup timed out after {self.startup_timeout} seconds."
        )
        return False
