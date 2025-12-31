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

from msmodelslim.ir.api import fake_quantize
from msmodelslim.ir.qal import QABCRegistry, QScheme, QScope, QDType, QParam, QStorage
from msmodelslim.utils.logging import logger_setter
from .auto import AutoFakeQuantActivation
from .const import int8_per_head_sym


@QABCRegistry.multi_register(
    dispatch_key=[
        int8_per_head_sym,
    ],
    abc_type=AutoFakeQuantActivation
)
@logger_setter()
class FakeQuantActivationPerHead(AutoFakeQuantActivation):
    """对称 per-head 伪量化/反量化。

    输入形状: (batch_size, num_head, seq_len, head_dim)，按第 1 维做 per-head。
    仅需 ext['scale']，忽略 offset。
    """

    def __init__(self, x_q_param: QParam):
        super().__init__()
        self.x_q_scheme = x_q_param.scheme

        scale = x_q_param.ext.get("scale")
        if scale is None:
            raise ValueError("`scale` is needed in ext but is missing for FakeQuantActivationPerHead")

        self.input_scale = nn.Parameter(scale, requires_grad=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 将 head 维（固定为索引 1）移动到最后一维，按每通道量化最后一维
        x_perm = x.movedim(1, -1)
        last_dim = x_perm.shape[-1]
        x_2d = x_perm.reshape(-1, last_dim)

        per_channel_scheme = QScheme(scope=QScope.PER_CHANNEL, dtype=self.x_q_scheme.dtype,
                                     symmetric=True)
        ext = {"scale": self.input_scale.data}

        x_q_param = QParam(scheme=per_channel_scheme, ext=ext)
        x_q_dq = fake_quantize(QStorage(QDType.FLOAT, x_2d), x_q_param)

        x_out = x_q_dq.value.reshape(x_perm.shape)
        x_out = x_out.movedim(-1, 1)
        return x_out
