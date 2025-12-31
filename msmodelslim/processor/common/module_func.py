#  -*- coding: utf-8 -*-
#  -*- coding: utf-8 -*-
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
from typing import Callable, Optional, Literal

from torch import nn

from msmodelslim.ir.qal import QABCRegistry
from msmodelslim.core.base.protocol import BatchProcessRequest
from msmodelslim.processor.base import AutoSessionProcessor, AutoProcessorConfig
from msmodelslim.utils.logging import get_logger, logger_setter

ModuleFuncType = Callable[[str, nn.Module], None]  # name, module


class ModuleFuncProcessorConfig(AutoProcessorConfig):
    type: Literal["module_func"] = "module_func"
    name: str
    func: ModuleFuncType


@QABCRegistry.register(dispatch_key=ModuleFuncProcessorConfig, abc_class=AutoSessionProcessor)
@logger_setter(prefix="msmodelslim.processor.module_func")
class ModuleFuncProcessor(AutoSessionProcessor):
    def __init__(
            self,
            model: nn.Module,
            config: ModuleFuncProcessorConfig,
            _: Optional[object] = None,
    ):
        super().__init__(model)
        self.config = config

    def __repr__(self) -> str:
        return f"ModuleFuncProcessor(name={self.config.name})"

    def preprocess(self, request: BatchProcessRequest) -> None:
        get_logger().debug(f"Running module func {self.config.name} on {request.name}")
        self.config.func(request.name, request.module)

    def is_data_free(self) -> bool:
        """
        判断处理器是否需要数据。
        
        Returns:
            True，因为加载处理器不需要输入数据
        """
        return True
