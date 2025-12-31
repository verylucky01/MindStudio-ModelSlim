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

from typing import Type, Tuple

import torch

from msmodelslim.ir.qal.qregistry import QFuncRegistry
from msmodelslim.ir.qal.qtypes import (
    Subgraph,
    NormLinearSubgraph,
)
from ..common import (
    SmoothQuantConfig,
    SmoothContext,
    IterSmoothScaleCalculator,
    SubgraphFusionFactory
)


@QFuncRegistry.register_api(dispatch_key=Tuple[Type[Subgraph], int])
def smooth_quant(subgraph: Subgraph, config: SmoothQuantConfig, context: SmoothContext) -> None:
    """
    使用smooth_quant算法进行异常值抑制
    """
    return QFuncRegistry.dispatch("smooth_quant",
                                  (type(subgraph), config.version),
                                  *(subgraph, config, context))


@torch.no_grad()
@QFuncRegistry.register(dispatch_key=(NormLinearSubgraph, 1), api_name="smooth_quant")
def smooth_quant_impl_norm_linear(subgraph: Subgraph, config: SmoothQuantConfig, context: SmoothContext) -> None:    
    # smooth_quant的scale计算逻辑同iter_smooth，仅scale_min固定为1e-5
    calculator = IterSmoothScaleCalculator(alpha=config.alpha, scale_min=1e-5)
    a_scale = context.a_smooth_scale
    w_scale = []
    for fc in subgraph.linears:
        fc_weight = fc.weight
        stat = fc_weight.abs().max(dim=0, keepdim=True)[0]
        w_scale.append(stat)
    w_scale = torch.cat(w_scale, dim=0)
    scales = calculator.compute_smooth_scale(a_scale, w_scale)
    shifts = {}
    if config.shift:
        linear_shifts = []
        for fc in subgraph.linears:
            linear_shift = torch.mm(
                context.shift.unsqueeze(0), fc.weight.data.clone().T
            ).squeeze(0)
            linear_shifts.append(linear_shift)
        shifts['linear_shifts'] = linear_shifts
        shifts['norm_shift'] = context.shift * -1 * (1.0 / scales)
        if subgraph.norm.bias is not None:
            subgraph.norm.bias.mul_(1.0 / scales)
    SubgraphFusionFactory.apply_fusion_to_subgraph(
        subgraph,
        scales={'scales': scales},
        shifts=shifts if shifts else None
    )
    return
