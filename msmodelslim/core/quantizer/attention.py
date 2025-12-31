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

import torch
from pydantic import validate_call
from torch import nn

from msmodelslim.ir.qal import QStorage
from msmodelslim.ir import AutoFakeQuantDynamicCache
from .base import AutoActQuantizer, QConfig


class DynamicCacheQuantizer(nn.Module):

    @validate_call(config=dict(arbitrary_types_allowed=True))
    def __init__(self, config: QConfig, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.config = config
        self.input_quantizer = AutoActQuantizer.from_config(config)

    @validate_call(config=dict(arbitrary_types_allowed=True))
    def setup(self):
        pass

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with QStorage.set_value_float_type(x.dtype):
            x = x.transpose(-2, -3)
            x_shape = x.shape
            x = x.reshape(-1, x.shape[-1] * x.shape[-2])
            dequant_x = self.input_quantizer(x)
            dequant_x = dequant_x.reshape(x_shape)
            dequant_x = dequant_x.transpose(-2, -3)
        return dequant_x

    def deploy(self):
        return AutoFakeQuantDynamicCache.create(
            self.input_quantizer.get_q_param(),
        )
