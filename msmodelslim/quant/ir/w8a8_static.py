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

from msmodelslim.core import QDType, QParam, QStorage
from msmodelslim.core import dequantize, fake_quantize
from msmodelslim.core.QAL import QABCRegistry, QScheme
from msmodelslim.core.QAL import QScope
from msmodelslim.quant.ir import AutoFakeQuantLinear
from msmodelslim.utils.logging import logger_setter
from .const import (
    int8_per_tensor_asym,
    int8_per_tensor_sym,
    int8_per_channel_sym
)


@QABCRegistry.multi_register(
    dispatch_key=[
        (int8_per_tensor_asym, int8_per_channel_sym),
        (int8_per_tensor_sym, int8_per_channel_sym),  # 也以非对称量化表示
    ],
    abc_type=AutoFakeQuantLinear
)
@logger_setter()
class W8A8StaticFakeQuantLinear(AutoFakeQuantLinear):
    """
    W8A8 per-channel/per-tensor sym/asym 静态非对称量化方式的伪量化IR。
    
    W8A8 静态非对称量化方式可以用以下参数描述：
        input_scale: 输入张量的量化参数，类型为torch.Tensor, dtype为torch.float32
        input_offset: 输入张量的量化参数，类型为torch.Tensor, dtype为torch.int32
        weight_scale: 权重张量的量化参数，类型为torch.Tensor, dtype为torch.float32
        weight: 权重张量，类型为torch.Tensor, dtype为torch.int8
        bias: 偏置张量，类型为torch.Tensor, dtype为torch.float32
    """

    def __init__(
            self,
            x_q_param: QParam,
            w_q_param: QParam,
            w_q: QStorage,
            bias: torch.Tensor
    ):
        super().__init__()

        self.input_scale = nn.Parameter(x_q_param.ext["scale"], requires_grad=False)
        self.input_offset = nn.Parameter(x_q_param.ext["offset"], requires_grad=False)
        self.weight_scale = nn.Parameter(w_q_param.ext["scale"], requires_grad=False)
        self.weight = nn.Parameter(w_q.value, requires_grad=False)
        self.bias = nn.Parameter(bias, requires_grad=False) if bias is not None else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_q_param = QParam(
            scheme=QScheme(scope=QScope.PER_TENSOR, dtype=QDType.INT8, symmetric=False),
            ext={"scale": self.input_scale.data, "offset": self.input_offset.data}
        )
        x_q_dq = fake_quantize(QStorage(QDType.FLOAT, x), x_q_param)

        w_q_param = QParam(
            scheme=QScheme(scope=QScope.PER_CHANNEL, dtype=QDType.INT8, symmetric=True),
            ext={"scale": self.weight_scale.data}
        )
        weight_q_dq = dequantize(QStorage(dtype=QDType.INT8, value=self.weight.data).T, w_q_param).T
        return F.linear(x_q_dq.value, weight_q_dq.value, self.bias)
