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
from torch import nn as nn
from torch.nn import functional as F

from msmodelslim.core import QParam, QStorage, calculate_qparam, QDType, fake_quantize, dequantize
from msmodelslim.core.QAL import QABCRegistry, QScope, QScheme
from msmodelslim.quant.ir import AutoFakeQuantLinear
from msmodelslim.quant.ir.const import fp8_e4m3_per_token_sym, fp8_e4m3_per_channel_sym
from msmodelslim.utils.logging import logger_setter
from msmodelslim.utils.exception import SchemaValidateError


@QABCRegistry.multi_register(
    dispatch_key=[
        (fp8_e4m3_per_token_sym, fp8_e4m3_per_channel_sym),
    ],
    abc_type=AutoFakeQuantLinear
)
@logger_setter()
class WFP8AFP8DynamicPerChannelFakeQuantLinear(AutoFakeQuantLinear):
    """
    W8A8FP8 per-channel/per-token sym/sym 动态对称量化方式的伪量化IR。
    """

    def __init__(
            self,
            x_q_param: QParam,
            w_q_param: QParam,
            w_q: QStorage,
            bias: torch.Tensor
    ):
        super().__init__()
        if "scale" not in w_q_param.ext:
            raise SchemaValidateError("scale is not in w_q_param.ext", action="Please check the w_q_param")
        self.weight_scale = nn.Parameter(w_q_param.ext["scale"].detach(), requires_grad=False)
        self.weight = nn.Parameter(w_q.value.detach(), requires_grad=False)
        self.bias = nn.Parameter(bias.detach(), requires_grad=False) if bias is not None else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_shape = x.shape
        x_reshape = x.reshape(-1, x_shape[-1])
        x_token_min = torch.amin(x_reshape, dim=0)
        x_token_max = torch.amax(x_reshape, dim=0)
        x_q_param = calculate_qparam(x_token_min, x_token_max, QDType.FP8_E4M3, QScope.PER_TOKEN, True)
        x_q_dq = fake_quantize(QStorage(QDType.FLOAT, x_reshape), x_q_param)
        w_q_param = QParam(scheme=QScheme(scope=QScope.PER_CHANNEL, dtype=QDType.FP8_E4M3, symmetric=True),
                           ext={"scale": self.weight_scale.data})
        weight_q_dq = dequantize(QStorage(dtype=QDType.FP8_E4M3, value=self.weight.data).T, w_q_param).T
        return F.linear(x_q_dq.value.reshape(x_shape), weight_q_dq.value, self.bias)
