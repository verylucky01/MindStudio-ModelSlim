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

from abc import abstractmethod

from msmodelslim.core.runner.pipeline_interface import PipelineInterface
from msmodelslim.utils.exception import ToDoError


class MultimodalPipelineInterface(PipelineInterface):
    """
    Interface for the multimodal pipeline inference.
    Multimodal has non transformer part, so we need to handle the non transformer part.
    """

    @abstractmethod
    def run_calib_inference(self):
        raise ToDoError(f"This model does not support run_calib_inference.",
                        action="Please implement run_calib_inference for your model.")

    @abstractmethod
    def apply_quantization(self, quant_model_func):
        """
        应用模型量化的抽象方法，子类需实现具体逻辑

        参数:
            quant_model_func: 量化函数（即api中的quant_model函数）
            quant_config: 量化配置对象
            calib_data: 校准数据
        """
        raise ToDoError(f"This model does not support apply_quantization.",
                        action="Please implement apply_quantization for your model.")

    @abstractmethod
    def load_pipeline(self):
        raise ToDoError(f"This model does not support load_pipeline.",
                        action="Please implement load_pipeline for your model.")

    @abstractmethod
    def set_model_args(self, override_model_config: object):
        raise ToDoError(f"This model does not support set_model_args.",
                        action="Please implement set_model_args for your model.")
