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
from torch import nn

from msmodelslim.core.KIA.manager import KIAManager
from msmodelslim.core.QAL.qregistry import QFuncRegistry
from msmodelslim.core.QAL.qtypes import (
    Subgraph,
    NormLinearSubgraph,
    LinearLinearSubgraph,
    OVSubgraph,
    UpDownSubgraph,
)
from msmodelslim.utils.logging import get_logger

from .alpha_beta_search import (
    FlexSmoothAlphaBetaSearcher,
    FlexAWQSSZAlphaBetaSearcher
)
from ..common import (
    FlexSmoothQuantConfig,
    FlexAWQSSZConfig,
    SmoothContext,
    validate_and_process_tensors,
    compute_weight_scale,
    compute_multi_weight_scale,
    FlexSmoothScaleCalculator,
    FlexAWQSSZScaleCalculator,
    SubgraphFusionFactory
)


@KIAManager.mark_require_version(min_version="1.0.0")
@QFuncRegistry.register_api(dispatch_key=Tuple[Type[Subgraph], int])
def flex_smooth_quant(subgraph: Subgraph, config: FlexSmoothQuantConfig, context: SmoothContext) -> None:
    """
    使用flex_smooth_quant算法进行异常值抑制
    
    Args:
        subgraph: 应用flex_smooth_quant算法的子图，支持以下类型：
            NormLinearSubgraph
            LinearLinearSubgraph
            OVSubgraph
            UpDownSubgraph
        config: FlexSmoothQuant算法配置
        context: 上下文，用于输入激活的smooth_scale，并记录权重的smooth_scale
        
    Returns:
        None: 无返回值
        
    """
    return QFuncRegistry.dispatch("flex_smooth_quant",
                                  (type(subgraph), config.version),
                                  *(subgraph, config, context))


@KIAManager.mark_require_version(min_version="1.0.0")
@QFuncRegistry.register_api(dispatch_key=Tuple[Type[Subgraph], int])
def flex_awq_ssz(subgraph: Subgraph, config: FlexAWQSSZConfig, context: SmoothContext) -> None:
    """
    使用flex_awq_ssz算法进行异常值抑制
    """
    return QFuncRegistry.dispatch("flex_awq_ssz",
                                  (type(subgraph), config.version),
                                  *(subgraph, config, context))


# ============== FlexSmoothQuant Implementation ==============

def get_optimal_alpha_beta_flex_smooth(
    config: FlexSmoothQuantConfig, act: torch.Tensor, fc_weights: torch.Tensor
) -> Tuple[float, float]:
    if config.alpha is None or config.beta is None:
        searcher = FlexSmoothAlphaBetaSearcher(act_sym=True, search_step=0.05)
        best_alpha, best_beta, final_mse = searcher.search_alpha_beta(act, fc_weights)
        get_logger().debug("Found optimal alpha: %.6f, beta: %.6f, final MSE: %.6f", best_alpha, best_beta, final_mse)
    else:
        best_alpha = config.alpha
        best_beta = config.beta
        get_logger().debug("Using provided alpha: %s, beta: %s", best_alpha, best_beta)
    return best_alpha, best_beta


@torch.no_grad()
@QFuncRegistry.register(dispatch_key=(OVSubgraph, 1), api_name="flex_smooth_quant")
def flex_smooth_impl_ov(subgraph: Subgraph, config: FlexSmoothQuantConfig, context: SmoothContext) -> None:
    tmp_device = next(subgraph.v_proj.parameters()).device
    dtype = subgraph.o_proj.weight.dtype
    act = validate_and_process_tensors(context, tmp_device, dtype)
    a_scale = context.a_smooth_scale
    fc_weights = subgraph.o_proj.weight

    best_alpha, best_beta = get_optimal_alpha_beta_flex_smooth(config, act, fc_weights)
    w_scale = compute_weight_scale(subgraph.o_proj.weight, dtype)

    group_method = 'mean'
    if config.extra_config is not None and config.extra_config.get('group_method') == 'max':
        group_method = 'max'
    
    calculator = FlexSmoothScaleCalculator(alpha=best_alpha, beta=best_beta, group_method=group_method)
    o_scales, v_scales = calculator.compute_ov_scales(
        a_scale, w_scale, 
        subgraph.num_attention_heads, 
        subgraph.key_value_heads
    )

    SubgraphFusionFactory.apply_fusion_to_subgraph(
        subgraph,
        scales={'o_scales': o_scales, 'v_scales': v_scales}
    )
    get_logger().debug("OV smoothing completed successfully")
    return


@torch.no_grad()
@QFuncRegistry.register(dispatch_key=(UpDownSubgraph, 1), api_name="flex_smooth_quant")
def flex_smooth_impl_up_down(subgraph: Subgraph, config: FlexSmoothQuantConfig, context: SmoothContext) -> None:
    tmp_device = next(subgraph.up_proj.parameters()).device
    dtype = subgraph.down_proj.weight.dtype
    act = validate_and_process_tensors(context, tmp_device, dtype)
    a_scale = context.a_smooth_scale
    fc_weights = subgraph.down_proj.weight
    best_alpha, best_beta = get_optimal_alpha_beta_flex_smooth(config, act, fc_weights)
    w_scale = compute_weight_scale(subgraph.down_proj.weight, dtype)
    calculator = FlexSmoothScaleCalculator(alpha=best_alpha, beta=best_beta)
    scales = calculator.compute_smooth_scale(a_scale, w_scale)
    SubgraphFusionFactory.apply_fusion_to_subgraph(
        subgraph,
        scales={'scales': scales}
    )
    get_logger().debug("Up-Down smoothing completed successfully")
    return


@torch.no_grad()
@QFuncRegistry.register(dispatch_key=(LinearLinearSubgraph, 1), api_name="flex_smooth_quant")
def flex_smooth_impl_linear_linear(
    subgraph: Subgraph, config: FlexSmoothQuantConfig, context: SmoothContext
) -> None:    
    tmp_device = next(subgraph.linear1.parameters()).device
    dtype = subgraph.linear2.weight.dtype
    act = validate_and_process_tensors(context, tmp_device, dtype)
    a_scale = context.a_smooth_scale
    fc_weights = subgraph.linear2.weight
    best_alpha, best_beta = get_optimal_alpha_beta_flex_smooth(config, act, fc_weights)
    w_scale = compute_weight_scale(subgraph.linear2.weight, dtype)
    calculator = FlexSmoothScaleCalculator(alpha=best_alpha, beta=best_beta)
    scales = calculator.compute_smooth_scale(a_scale, w_scale)
    SubgraphFusionFactory.apply_fusion_to_subgraph(
        subgraph,
        scales={'scales': scales}
    )
    get_logger().debug("Linear-Linear smoothing completed successfully")
    return


@torch.no_grad()
@QFuncRegistry.register(dispatch_key=(NormLinearSubgraph, 1), api_name="flex_smooth_quant")
def flex_smooth_impl_norm_linear(subgraph: Subgraph, config: FlexSmoothQuantConfig, context: SmoothContext) -> None:
    tmp_device = next(subgraph.norm.parameters()).device
    dtype = subgraph.linears[0].weight.dtype
    act = validate_and_process_tensors(context, tmp_device, dtype)
    a_scale = context.a_smooth_scale
    fc_weights = torch.cat([fc.weight for fc in subgraph.linears], dim=0)
    best_alpha, best_beta = get_optimal_alpha_beta_flex_smooth(config, act, fc_weights)
    w_scale = compute_multi_weight_scale([fc.weight for fc in subgraph.linears], dtype)
    calculator = FlexSmoothScaleCalculator(alpha=best_alpha, beta=best_beta)
    scales = calculator.compute_smooth_scale(a_scale, w_scale)
    SubgraphFusionFactory.apply_fusion_to_subgraph(
        subgraph,
        scales={'scales': scales}
    )
    get_logger().debug("Norm-Linear smoothing completed successfully")
    return


# ============== FlexAWQSSZ Implementation ==============

@torch.no_grad()
def get_optimal_alpha_beta(config: FlexAWQSSZConfig, act, linear: nn.Linear):
    if config.alpha is None:
        searcher = FlexAWQSSZAlphaBetaSearcher(qconfig=config.qconfig)
        best_alpha, normal_mse_best = searcher.search_alpha(act, linear)
        get_logger().debug(f"Found optimal alpha: {best_alpha:.6f}, MSE: {normal_mse_best:.6f}")
    else:
        best_alpha = config.alpha
        get_logger().debug(f"Using provided alpha: {best_alpha}")
    best_beta = 0
    return best_alpha, best_beta


@torch.no_grad()
@QFuncRegistry.register(dispatch_key=(OVSubgraph, 1), api_name="flex_awq_ssz")
def flex_awq_ssz_impl_ov(
    subgraph: Subgraph, config: FlexAWQSSZConfig, context: SmoothContext
) -> None:   
    tmp_device = next(subgraph.v_proj.parameters()).device
    dtype = subgraph.o_proj.weight.dtype
    act = validate_and_process_tensors(context, tmp_device, dtype)
    # flex awq ssz 的 act 尺度计算使用mean
    act_scales = torch.mean(torch.abs(act), dim=0, keepdim=True)[0]
    best_alpha, best_beta = get_optimal_alpha_beta(config, act, subgraph.o_proj)
    
    w_scale = compute_weight_scale(subgraph.o_proj.weight, dtype)
    calculator = FlexAWQSSZScaleCalculator(alpha=best_alpha, beta=best_beta)
    o_scales, v_scales = calculator.compute_ov_scales(
        act_scales, w_scale,
        subgraph.num_attention_heads,
        subgraph.key_value_heads
    )
    
    SubgraphFusionFactory.apply_fusion_to_subgraph(
        subgraph,
        scales={'o_scales': o_scales, 'v_scales': v_scales}
    )
    get_logger().debug("OV smoothing completed successfully")
    return


@torch.no_grad()
@QFuncRegistry.register(dispatch_key=(UpDownSubgraph, 1), api_name="flex_awq_ssz")
def flex_awq_ssz_impl_up_down(
    subgraph: Subgraph, config: FlexAWQSSZConfig, context: SmoothContext
) -> None:
    tmp_device = next(subgraph.up_proj.parameters()).device
    dtype = subgraph.down_proj.weight.dtype
    act = validate_and_process_tensors(context, tmp_device, dtype)
    act_scales = torch.mean(torch.abs(act), dim=0, keepdim=True)[0]
    best_alpha, best_beta = get_optimal_alpha_beta(config, act, subgraph.down_proj)
    w_scale = compute_weight_scale(subgraph.down_proj.weight, dtype)
    calculator = FlexAWQSSZScaleCalculator(alpha=best_alpha, beta=best_beta)
    scales = calculator.compute_smooth_scale(act_scales, w_scale)
    SubgraphFusionFactory.apply_fusion_to_subgraph(
        subgraph,
        scales={'scales': scales}
    )
    get_logger().debug("Up-Down smoothing completed successfully")
    return


@torch.no_grad()
@QFuncRegistry.register(dispatch_key=(LinearLinearSubgraph, 1), api_name="flex_awq_ssz")
def flex_awq_ssz_impl_linear_linear(
    subgraph: Subgraph, config: FlexAWQSSZConfig, context: SmoothContext
) -> None:
    tmp_device = next(subgraph.linear1.parameters()).device
    dtype = subgraph.linear2.weight.dtype
    act = validate_and_process_tensors(context, tmp_device, dtype)
    act_scales = torch.mean(torch.abs(act), dim=0, keepdim=True)[0]
    best_alpha, best_beta = get_optimal_alpha_beta(config, act, subgraph.linear2)
    w_scale = compute_weight_scale(subgraph.linear2.weight, dtype)
    calculator = FlexAWQSSZScaleCalculator(alpha=best_alpha, beta=best_beta)
    scales = calculator.compute_smooth_scale(act_scales, w_scale)
    SubgraphFusionFactory.apply_fusion_to_subgraph(
        subgraph,
        scales={'scales': scales}
    )
    get_logger().debug("Linear-Linear smoothing completed successfully")
    return


@torch.no_grad()
@QFuncRegistry.register(dispatch_key=(NormLinearSubgraph, 1), api_name="flex_awq_ssz")
def flex_awq_ssz_impl_norm_linear(
    subgraph: Subgraph, config: FlexAWQSSZConfig, context: SmoothContext
) -> None:
    tmp_device = next(subgraph.norm.parameters()).device
    dtype = subgraph.linears[0].weight.dtype
    act = validate_and_process_tensors(context, tmp_device, dtype)
    act_scales = torch.mean(torch.abs(act), dim=0, keepdim=True)[0]
    # 仅用前两层做校准
    if len(subgraph.linears) > 3:
        fc_weights = torch.cat([fc.weight for fc in subgraph.linears[0:2]], dim=0)
    else:
        fc_weights = torch.cat([fc.weight for fc in subgraph.linears], dim=0)
    
    get_logger().debug(
        "Activation scale shape: %s, Weight shape: %s",
        act_scales.shape, fc_weights.shape
    )
    merged_linear = nn.Linear(
        in_features=fc_weights.shape[1],
        out_features=fc_weights.shape[0],
        bias=False,
        device=fc_weights.device,
        dtype=fc_weights.dtype
    )
    merged_linear.weight.data = fc_weights
    best_alpha, best_beta = get_optimal_alpha_beta(config, act, merged_linear)

    weights = [fc.weight for fc in subgraph.linears]
    w_scale = compute_multi_weight_scale(weights, dtype)
    calculator = FlexAWQSSZScaleCalculator(alpha=best_alpha, beta=best_beta)
    scales = calculator.compute_smooth_scale(act_scales, w_scale)
    get_logger().debug("Computed smooth scales shape: %s", scales.shape)
    SubgraphFusionFactory.apply_fusion_to_subgraph(
        subgraph,
        scales={'scales': scales}
    )
    get_logger().debug("Norm-Linear smoothing completed successfully")
    return
