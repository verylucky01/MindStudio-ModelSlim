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
from typing import Any, List

from torch import nn

from msmodelslim.core.const import DeviceType
from msmodelslim.model import IModel
from msmodelslim.utils.exception import UnsupportedError


class PipelineInterface(IModel):
    """
    Interface for determining the pipeline of model inference.
    ModelSlim V0 is a simple quant service, which does NOT schedule the model inference pipeline in finer granularity.
    Just show how to handle dataset and load model. Remaining parts are left to the model itself.
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
        raise UnsupportedError(
            "This model does not support generate dataset.",
            action="Please implement generate_dataset in PipelineInterface.")

    @abstractmethod
    def load_model(self, device: DeviceType = DeviceType.NPU) -> nn.Module:
        """
        Load the model to specified device.
        After loading, the model should be ready for inference.

        Returns:
            nn.Module: The loaded model.
        """
        raise UnsupportedError(
            "This model does not support load model to specified device and torch dtype.",
            action="Please implement load_model in PipelineInterface.")
