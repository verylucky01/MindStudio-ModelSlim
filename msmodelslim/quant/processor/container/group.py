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

from typing import Optional, Literal

from torch import nn

from msmodelslim.core.QAL import QABCRegistry
from msmodelslim.core.base.protocol import BatchProcessRequest
from msmodelslim.quant.processor.base import AutoSessionProcessor, AutoProcessorConfig, AutoProcessorConfigList


class GroupProcessorConfig(AutoProcessorConfig):
    type: Literal['group']
    configs: AutoProcessorConfigList


@QABCRegistry.register(dispatch_key=GroupProcessorConfig, abc_class=AutoSessionProcessor)
class GroupProcessor(AutoSessionProcessor):
    """
    前向量化处理器合并器，用于将多个前向量化处理器合并为一个处理器，用于减少模型推理的次数。
    """

    def __init__(
            self,
            model: nn.Module,
            config: GroupProcessorConfig,
            adapter: Optional[object] = None,
    ):
        super().__init__(model)
        self.processors = [AutoSessionProcessor.from_config(model, cfg, adapter) for cfg in config.configs]
        self.processor_names = [processor.__class__.__name__ for processor in self.processors]

    def __repr__(self) -> str:
        return f"GroupProcessor(processors={self.processor_names})"

    def preprocess(self, request: BatchProcessRequest) -> None:
        for processor in self.processors:
            processor.preprocess(request)

    def postprocess(self, request: BatchProcessRequest) -> None:
        for processor in self.processors:
            processor.postprocess(request)

    def pre_run(self) -> None:
        for processor in self.processors:
            processor.pre_run()

    def post_run(self) -> None:
        for processor in self.processors:
            processor.post_run()

    def is_data_free(self) -> bool:
        """
        判断处理器是否需要数据。
        """
        return all(processor.is_data_free() for processor in self.processors)

    def need_kv_cache(self):
        return any(processor.need_kv_cache() for processor in self.processors)
    
    def support_distributed(self) -> bool:
        return any(processor.support_distributed() for processor in self.processors)
