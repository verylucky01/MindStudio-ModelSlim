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
import torch.nn as nn
import torch.nn.functional as F

from msmodelslim.core import QDType, QParam, QStorage, calculate_qparam
from msmodelslim.core import dequantize, fake_quantize
from msmodelslim.core.QAL import QABCRegistry, QScheme
from msmodelslim.core.QAL import QScope
from msmodelslim.quant.ir import AutoFakeQuantLinear
from msmodelslim.utils.logging import logger_setter
from .const import (
    int8_per_token_sym,
    int4_per_channel_sym
)


@QABCRegistry.multi_register(
    dispatch_key=[
        (int8_per_token_sym, int4_per_channel_sym),
    ],
    abc_type=AutoFakeQuantLinear
)
@logger_setter()
class W4A8DynamicFakeQuantLinear(AutoFakeQuantLinear):
    def __init__(
            self,
            x_q_param: QParam,
            w_q_param: QParam,
            w_q: QStorage,
            bias: torch.Tensor
    ):
        super().__init__()
        self.weight_scale = nn.Parameter(w_q_param.ext["scale"].detach(), requires_grad=False)
        self.weight = nn.Parameter(w_q.value.detach(), requires_grad=False)
        self.bias = nn.Parameter(bias.detach(), requires_grad=False) if bias is not None else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_shape = x.shape
        x_reshape = x.reshape(-1, x_shape[-1])
        x_token_min = torch.amin(x_reshape, dim=0)
        x_token_max = torch.amax(x_reshape, dim=0)
        x_q_param = calculate_qparam(x_token_min, x_token_max, QDType.INT8, QScope.PER_TOKEN, True)
        x_q_dq = fake_quantize(QStorage(QDType.FLOAT, x_reshape), x_q_param)
        w_q_param = QParam(scheme=QScheme(scope=QScope.PER_CHANNEL, dtype=QDType.INT4, symmetric=True),
                           ext={"scale": self.weight_scale.data})
        weight_q_dq = dequantize(QStorage(dtype=QDType.INT4, value=self.weight.data).T, w_q_param).T
        return F.linear(x_q_dq.value.reshape(x_shape), weight_q_dq.value, self.bias)
