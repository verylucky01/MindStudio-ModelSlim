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
from typing import Generator, Any, List

from torch import nn

from msmodelslim.core.base.protocol import ProcessRequest
from msmodelslim.core.const import DeviceType
from msmodelslim.model import IModel
from msmodelslim.utils.exception import ToDoError


class PipelineInterface(IModel):
    """
    Interface for determining the pipeline of model inference.
    Runner schedules the process of model in finer granularity.
    The granularity of the pipeline is determined by users via the implementation of this interface.
    """

    @abstractmethod
    def handle_dataset(self, dataset: Any, device: DeviceType = DeviceType.NPU) -> List[Any]:
        """
        Handle the dataset for model inference.
        The dataset should be converted into a List of data
            that can be directly used for model inference(model(*data) or model(**data)).
        Returns:
            List[Any]: The processed dataset.
        """
        raise ToDoError(
            "This model does not support generate dataset.",
            action="Please implement generate_dataset for PipelineInterface.")

    @abstractmethod
    def init_model(self, device: DeviceType = DeviceType.NPU) -> nn.Module:
        """
        Init the model to specified device which may be different from execution device.
        If the model is large, just load a part of the model.
        You can extend the model when generating model visit and forward.
        
        Returns:
            nn.Module: The loaded model.
        """
        raise ToDoError(
            "This model does not support init model to specified device.",
            action="Please implement init_model for PipelineInterface.")

    @abstractmethod
    def generate_model_visit(self, model: nn.Module) -> Generator[ProcessRequest, Any, None]:
        """
        Determine the model visit, which is used to modify the model in fine-scheduling runner.
        The model visit pipeline is a generator of ProcessRequest,
            which decomposes the model modification into a list of module visit.
        NOTICE: The yield sequence of modules in ProcessRequest should be same as generate_model_forward.

        Returns:
            Generator[ProcessRequest, Any, None]: The generator of model visit.
        """
        raise ToDoError(
            "This model does not support generate_model_visit, which is required for model modification.",
            action="Please implement generate_model_visit for PipelineInterface.")

    @abstractmethod
    def generate_model_forward(self, model: nn.Module, inputs: Any,
                               ) -> Generator[ProcessRequest, Any, None]:
        """
        Determine the model forward, which is used to calibrate the model in fine-scheduling runner.
        The model forward pipeline is a generator of ProcessRequest,
            which decomposes the model forward into a list of module forward.
        NOTICE: The yield sequence of modules in ProcessRequest should be same as generate_model_visit.

        Returns:
            Generator[ProcessRequest, Any, None]: The generator of model forward.
        """
        raise ToDoError(
            "This model does not support generate_model_forward, which is required for model calibration.",
            action="Please implement generate_model_forward for PipelineInterface.")

    @abstractmethod
    def enable_kv_cache(self, model: nn.Module, need_kv_cache: bool) -> None:
        """
        Enable/Disable the kv cache for the model.
        Quantization Calibration probably does NOT requires the kv cache, which depends on specific processor.
        Disable kv cache can reduce the memory usage.
        """
        raise ToDoError(
            "This model does not support enable_kv_cache, which is required for model calibration.",
            action="Please implement enable_kv_cache for PipelineInterface.")
