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
    LinearLinearSubgraph,
    OVSubgraph,
    UpDownSubgraph,
)
from ..common import (
    IterSmoothConfig,
    SmoothContext,
    IterSmoothScaleCalculator,
    SubgraphFusionFactory
)


@QFuncRegistry.register_api(dispatch_key=Tuple[Type[Subgraph], int])
def iter_smooth(subgraph: Subgraph, config: IterSmoothConfig, context: SmoothContext) -> None:
    """
    使用iter_smooth算法进行异常值抑制
    
    Args:
        subgraph: 应用iter_smooth算法的子图，支持以下类型：
            NormLinearSubgraph
            LinearLinearSubgraph
            OVSubgraph
            UpDownSubgraph
        config: IterSmooth算法配置
        context: 上下文，用于输入激活的smooth_scale，并记录权重的smooth_scale
        
    Returns:
        None: 无返回值
        
    """
    return QFuncRegistry.dispatch("iter_smooth",
                                  (type(subgraph), config.version),
                                  *(subgraph, config, context))


@torch.no_grad()
@QFuncRegistry.register(dispatch_key=(OVSubgraph, 1), api_name="iter_smooth")
def iter_smooth_impl_ov(subgraph: Subgraph, config: IterSmoothConfig, context: SmoothContext) -> None:
    calculator = IterSmoothScaleCalculator(alpha=config.alpha, scale_min=config.scale_min)
    a_scale = context.a_smooth_scale
    w_scale = subgraph.o_proj.weight
    scales = calculator.compute_smooth_scale(a_scale, w_scale)
    o_scales, v_scales = calculator.compute_ov_scales(
        a_scale, w_scale, subgraph.num_attention_heads, subgraph.key_value_heads
    )
    shifts = {}
    if config.shift:
        shifts['o_shift'] = torch.mm(
            context.shift.unsqueeze(0), subgraph.o_proj.weight.data.clone().T
        ).squeeze(0)
        shifts['v_shift'] = context.shift * -1 * (1.0 / scales)
        if subgraph.v_proj.bias is not None:
            subgraph.v_proj.bias.mul_(1.0 / scales)
    SubgraphFusionFactory.apply_fusion_to_subgraph(
        subgraph,
        scales={'o_scales': o_scales, 'v_scales': v_scales},
        shifts=shifts if shifts else None
    )
    return


@torch.no_grad()
@QFuncRegistry.register(dispatch_key=(UpDownSubgraph, 1), api_name="iter_smooth")
def iter_smooth_impl_up_down(subgraph: Subgraph, config: IterSmoothConfig, context: SmoothContext) -> None:
    calculator = IterSmoothScaleCalculator(alpha=config.alpha, scale_min=config.scale_min)
    a_scale = context.a_smooth_scale
    w_scale = subgraph.down_proj.weight
    scales = calculator.compute_smooth_scale(a_scale, w_scale)
    shifts = {}
    if config.shift:
        shifts['down_shift'] = torch.mm(
            context.shift.unsqueeze(0), subgraph.down_proj.weight.data.clone().T
        ).squeeze(0)
        shifts['up_shift'] = context.shift * -1 * (1.0 / scales)
        if subgraph.up_proj.bias is not None:
            subgraph.up_proj.bias.mul_(1.0 / scales)
    SubgraphFusionFactory.apply_fusion_to_subgraph(
        subgraph,
        scales={'scales': scales},
        shifts=shifts if shifts else None
    )
    return


@torch.no_grad()
@QFuncRegistry.register(dispatch_key=(LinearLinearSubgraph, 1), api_name="iter_smooth")
def iter_smooth_impl_linear_linear(subgraph: Subgraph, config: IterSmoothConfig, context: SmoothContext) -> None:
    calculator = IterSmoothScaleCalculator(alpha=config.alpha, scale_min=config.scale_min)
    a_scale = context.a_smooth_scale
    w_scale = subgraph.linear2.weight
    scales = calculator.compute_smooth_scale(a_scale, w_scale)
    shifts = {}
    if config.shift:
        shifts['linear2_shift'] = torch.mm(
            context.shift.unsqueeze(0), subgraph.linear2.weight.data.clone().T
        ).squeeze(0)
        shifts['linear1_shift'] = context.shift * -1 * (1.0 / scales)
        if subgraph.linear1.bias is not None:
            subgraph.linear1.bias.mul_(1.0 / scales)
    SubgraphFusionFactory.apply_fusion_to_subgraph(
        subgraph,
        scales={'scales': scales},
        shifts=shifts if shifts else None
    )
    return


@torch.no_grad()
@QFuncRegistry.register(dispatch_key=(NormLinearSubgraph, 1), api_name="iter_smooth")
def iter_smooth_impl_norm_linear(subgraph: Subgraph, config: IterSmoothConfig, context: SmoothContext) -> None:    
    calculator = IterSmoothScaleCalculator(alpha=config.alpha, scale_min=config.scale_min)
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
