#!/usr/bin/env python
# -*- coding: UTF-8 -*-

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

from typing import List, Optional, Literal

from pydantic import Field, ConfigDict
from torch import nn
from torch import distributed as dist

from msmodelslim.ir.qal import QScope, QDType
from msmodelslim.ir.qal.qregistry import QABCRegistry
from msmodelslim.core.base.protocol import BatchProcessRequest
from msmodelslim.processor.base import AutoSessionProcessor, AutoProcessorConfig
from msmodelslim.core.quantizer.linear import LinearQuantizer, LinearQConfig
from msmodelslim.utils.config_map import ConfigSet
from msmodelslim.utils.logging import get_logger, logger_setter
from msmodelslim.utils.distributed import DistHelper


class LinearProcessorConfig(AutoProcessorConfig):
    type: Literal["linear_quant"] = "linear_quant"
    qconfig: LinearQConfig = Field(description="量化配置")
    include: List[str] = Field(default_factory=lambda: ["*"], description="包含的模块名称")
    exclude: List[str] = Field(default_factory=lambda: [], description="排除的模块名称")

    model_config = ConfigDict(extra="forbid")


def _warning_unmatched_pattern(name: str, config_set: ConfigSet) -> None:
    unmatched_keys = config_set.unmatched_keys()
    unmatched_keys = list(filter(lambda x: x != "*", unmatched_keys))
    if unmatched_keys:
        get_logger().warning(
            f"These {name} patterns are not matched any module, please ensure this is as expected: {unmatched_keys}")


@QABCRegistry.register(dispatch_key=LinearProcessorConfig, abc_class=AutoSessionProcessor)
@logger_setter(prefix="msmodelslim.processor.linear_quant")
class LinearQuantProcessor(AutoSessionProcessor):
    def __init__(
            self,
            model: nn.Module,
            config: LinearProcessorConfig,
            adapter: Optional[object] = None,
    ):
        super().__init__(model)
        self.config = config
        self.include = ConfigSet(config.include)
        self.exclude = ConfigSet(config.exclude)
        
        self.dist_helper = None

    def is_data_free(self) -> bool:
        """
        判断是否是data free场景
        通过检查 LinearQuantizer 是否data free场景
        
        Returns:
            bool: 是否data free场景
        """
        return LinearQuantizer(self.config.qconfig).is_data_free()

    def support_distributed(self) -> bool:
        """
        判断是否支持分布式
        通过检查 LinearQuantizer 是否支持分布式来判断
        
        Returns:
            bool: 是否支持分布式
        """
        # 创建一个临时的 LinearQuantizer 实例来检查是否支持分布式
        temp_quantizer = LinearQuantizer(self.config.qconfig)
        return temp_quantizer.support_distributed()

    def post_run(self) -> None:
        _warning_unmatched_pattern("include", self.include)
        _warning_unmatched_pattern("exclude", self.exclude)

    def preprocess(self, request: BatchProcessRequest) -> None:
        # 在preprocess时创建DistHelper，传入prefix信息
        if dist.is_initialized():
            self.dist_helper = DistHelper(request.module, prefix=request.name)
        self._install_quantizer(request.name, request.module)

    def postprocess(self, request: BatchProcessRequest) -> None:
        self._deploy(request.name, request.module)
        # 清理分布式辅助类
        self.dist_helper = None

    def _install_quantizer(self, prefix: str, module: nn.Module) -> None:
        for name, submodule in module.named_modules(prefix=prefix):

            if not isinstance(submodule, nn.Linear):
                continue

            if name not in self.include:
                continue

            if name in self.exclude:
                continue

            self._process_linear(name, submodule)

    def _deploy(self, prefix: str, module: nn.Module) -> None:
        for name, submodule in module.named_modules(prefix=prefix):
            if hasattr(submodule, "deploy"):
                self.model.set_submodule(name, submodule.deploy())

    def _process_linear(self, full_name: str, module: nn.Linear) -> None:
        """
        处理线性层，判断是否需要启用同步操作
        
        同步操作的启用条件：
        1. 分布式已启动 (dist.is_initialized())
        2. 该模块是共享的 (is_shared)
        
        Args:
            full_name: 模块全名
            module: 线性层模块
        """
        quantizer = LinearQuantizer(self.config.qconfig)
        
        # 判断是否需要启用同步操作
        if self.dist_helper is not None and self.dist_helper.is_shared(full_name):
            quantizer.enable_sync()
        
        quantizer.setup(module)
        self.model.set_submodule(full_name, quantizer)  
