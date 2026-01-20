"""
-------------------------------------------------------------------------
This file is part of the MindStudio project.
Copyright (c) 2025 Huawei Technologies Co.,Ltd.

MindStudio is licensed under Mulan PSL v2.
You can use this software according to the terms and conditions of the Mulan PSL v2.
You may obtain a copy of Mulan PSL v2 at:

         http://license.coscl.org.cn/MulanPSL2

THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
See the Mulan PSL v2 for more details.
-------------------------------------------------------------------------
"""
from abc import ABC, abstractmethod
from typing import List, Optional

from pydantic import Field

from msmodelslim.core.tune_strategy import EvaluateAccuracy
from msmodelslim.utils.plugin import TypedConfig
from msmodelslim.utils.security import build_safe_url, safe_post

PRECHECK_CONFIG_PLUGIN_PATH = "msmodelslim.precheck_config.plugins"


@TypedConfig.plugin_entry(entry_point_group=PRECHECK_CONFIG_PLUGIN_PATH)
class BasePrecheckConfig(TypedConfig):
    """预检查配置基类"""
    type: TypedConfig.TypeField  # 类型字段，用于插件注册
    max_tokens: int = Field(default=512, gt=0, description="Maximum number of tokens to generate, must be greater than 0")
    timeout: float = Field(default=60.0, gt=0.0, description="Request timeout in seconds for API calls, must be greater than 0")


class BasePrecheckRule(ABC):
    """
    预检查规则基类。
    
    包含公共功能：执行单轮对话、进行特殊预检动作等。
    """
    
    def __init__(self, config: BasePrecheckConfig):
        """
        初始化预检查规则。
        
        Args:
            config: 预检查配置（可以是 BasePrecheckConfig 或其子类）
        """
        self.config = config
    
    def test_chat_via_api(
        self,
        host: str,
        port: int,
        served_model_name: str,
        test_message: str,
        max_tokens: Optional[int] = None
    ) -> str:
        """
        通过 HTTP API 发送测试消息。
        
        Args:
            host: 服务器主机地址
            port: 服务器端口
            served_model_name: 模型名称
            test_message: 测试消息
            max_tokens: 最大生成 token 数，如果为 None 则使用配置中的值
            
        Returns:
            模型的响应文本
            
        Raises:
            Exception: 如果请求失败
        """
        chat_url = build_safe_url(
            host=host,
            port=port,
            endpoint="/v1/chat/completions",
            scheme='http'
        )
        
        payload = {
            "model": served_model_name,
            "messages": [
                {"role": "user", "content": test_message}
            ],
            "max_tokens": max_tokens or self.config.max_tokens,
            "temperature": 0.1,
            "extra_body": {
                "chat_template_kwargs": {
                    "thinking": False,
                }
            }
        }
        
        response = safe_post(
            url=chat_url,
            json=payload,
            timeout=self.config.timeout
        )
        response.raise_for_status()
        result = response.json()
        
        # 提取响应内容
        if "choices" in result and len(result["choices"]) > 0:
            message = result["choices"][0].get("message", {})
            content = message.get("content", "")
            return content
        else:
            raise ValueError("Invalid response format from chat API")
    
    @abstractmethod
    def check(
        self,
        host: str,
        port: int,
        served_model_name: str,
        datasets: List[str]
    ) -> Optional[List[EvaluateAccuracy]]:
        """
        执行预检查。
        
        Args:
            host: 服务器主机地址
            port: 服务器端口
            served_model_name: 模型名称
            datasets: 数据集列表
            
        Returns:
            如果检查失败，返回不达标的结果列表；如果检查通过，返回 None
        """
        pass
