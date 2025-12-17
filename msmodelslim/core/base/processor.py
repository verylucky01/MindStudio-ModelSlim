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

from torch import nn

from msmodelslim.core.base.protocol import BatchProcessRequest


class BaseProcessor:
    """
    处理器的基类，用于处理模型推理过程中的特定事件。
    
    处理器负责在模型推理的不同阶段执行特定的操作，如数据预处理、后处理等。
    所有具体的处理器实现都应该继承此类并实现抽象方法。
    
    """

    def __init__(self, model: nn.Module):
        self.model = model

    def process(self, request: BatchProcessRequest) -> None:
        """
        处理请求并返回响应。
        
        该方法根据请求中的事件类型、目标模块和相关数据，执行相应的处理操作，
        并返回处理结果和控制选项，指导后续处理流程。
        
        参数:
            request: 处理请求，包含事件类型、目标模块和相关数据
        """
        self._run_forward_if_need(request)

    def pre_run(self) -> None:
        """
        Runner进行调度前，执行的操作。
        """
        pass

    def post_run(self) -> None:
        """
        Runner结束调度后，执行的操作。
        """
        pass

    def preprocess(self, request: BatchProcessRequest) -> None:
        """
        在Runner调度该Processor的process前，执行的操作。
        """
        pass

    def postprocess(self, request: BatchProcessRequest) -> None:
        """
        在Runner调度该Processor的process后，执行的操作。
        """
        pass

    def _run_forward_if_need(self, request: BatchProcessRequest) -> None:
        _ = self
        outputs = []
        for data in request.datas:
            args, kwargs = data
            if args or kwargs:
                output = request.module(*args, **kwargs)
                outputs.append(output)
        request.outputs = outputs
