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


from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
import torch.nn as nn

from msmodelslim.utils.exception import MisbehaviorError
from msmodelslim.utils.logging import get_logger

from .smooth_types import SmoothContext


@torch.no_grad()
def validate_and_process_tensors(context: SmoothContext, tmp_device, dtype) -> torch.Tensor:
    valid_tensors = [
        tensor.to(dtype=dtype).to(tmp_device)
        for tensor in context.tensors
        if tensor is not None and torch.is_tensor(tensor)
    ]
    tensors = torch.cat(valid_tensors, dim=0)
    act = tensors.view(-1, tensors.shape[-1])
    return act


@dataclass
class MQGAScaleParams:
    act_scales: torch.Tensor
    weight_scales: torch.Tensor
    best_alpha: float
    best_beta: float
    num_key_value_groups: int
    head_dim: int


@torch.no_grad()
def compute_weight_scale(weight: torch.Tensor, dtype) -> torch.Tensor:
    weight_abs = weight.abs()
    weight_max = weight_abs.max(dim=0, keepdim=True)[0]
    w_scale = weight_max.max(dim=0)[0]
    w_scale = w_scale.to(torch.float32).clamp(min=1e-5).to(dtype=dtype)
    get_logger().debug(
        "Weight scale shape: %s, min: %.6f, max: %.6f",
        w_scale.shape, w_scale.min(), w_scale.max()
    )
    return w_scale


@torch.no_grad()
def compute_multi_weight_scale(weights: List[torch.Tensor], dtype) -> torch.Tensor:
    weight_stat = [w.abs().max(dim=0, keepdim=True)[0] for w in weights]
    w_scale = torch.cat(weight_stat, dim=0)
    w_scale = w_scale.max(dim=0)[0].to(torch.float32).clamp(min=1e-5).to(dtype=dtype)
    get_logger().debug(
        "Multi-weight scale shape: %s, min: %.6f, max: %.6f",
        w_scale.shape, w_scale.min(), w_scale.max()
    )
    return w_scale


@torch.no_grad()
def apply_smooth_scale_shift(layer, scales, shift=None):
    device = layer.weight.device
    dtype = layer.weight.dtype
    layer.weight.mul_(scales)
    
    if shift is not None:
        shift = shift.to(device).to(dtype)
        if layer.bias is None:
            bias_shape = (layer.weight.shape[0],)
            layer.bias = nn.Parameter(torch.zeros(bias_shape, device=device, dtype=dtype))
        
        layer.bias.add_(shift)


@torch.no_grad()
def prepare_mqga_parameters(num_attention_heads: int, num_key_value_heads: int) -> Tuple[int, int]:
    ratio = num_attention_heads // num_key_value_heads
    scales_pad_size = 0
    return ratio, scales_pad_size


@torch.no_grad()
def reduce_scales_for_mqga_mean(
    scales: torch.Tensor, shape_ratio: int, num_attention_heads: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Reduce MQGA scales using mean method
    
    Assumes K V activation heads are broadcast in the following pattern:
    [h1, h2, h3, h4] -> [h1, ... , h1, h2,..., h2, h3, ..., h3, h4, ..., h4]
    
    Args:
        scales: Scale tensor
        shape_ratio: Shape ratio
        num_attention_heads: Number of attention heads
    
    Returns:
        Tuple of (updated_scales, reduced_scales)
    """
    num_q_heads = num_attention_heads
    
    # Validate divisibility: num_q_heads must be divisible by shape_ratio
    if num_q_heads % shape_ratio != 0:
        raise MisbehaviorError(
            f"num_attention_heads ({num_q_heads}) must be divisible by shape_ratio ({shape_ratio}), "
            f"but got remainder {num_q_heads % shape_ratio}",
            action="Please check the model configuration to ensure num_attention_heads is divisible by shape_ratio"
        )
    
    num_kv_heads = num_q_heads // shape_ratio
    head_emb_size = scales.size(0) // num_q_heads
    reduced_scales, updated_scales = [], []
    copied_scales_slices = scales.split(scales.size(0) // num_kv_heads)
    
    for gr_idx in range(num_kv_heads):
        subset_of_scales = copied_scales_slices[gr_idx].view(-1, head_emb_size)
        repeat_num = subset_of_scales.size(0)
        reduced_scales.append(subset_of_scales.mean(0))
        updated_scales.append(torch.cat([reduced_scales[-1]] * repeat_num))
    return torch.cat(updated_scales), torch.cat(reduced_scales)


@torch.no_grad()
def reduce_scales_for_mqga_max(params: MQGAScaleParams) -> Tuple[torch.Tensor, torch.Tensor]:
    """Reduce MQGA scales using max method
    
    Args:
        params: MQGA scale parameters
    
    Returns:
        Tuple of (o_scales, v_scales)
    """
    act_scales = params.act_scales.view(-1, params.num_key_value_groups, params.head_dim).max(dim=1)[0]
    weight_scales = params.weight_scales.view(-1, params.num_key_value_groups, params.head_dim).max(dim=1)[0]
    group_scales = (act_scales.pow(params.best_alpha) /
                    weight_scales.pow(params.best_beta)).to(torch.float32).clamp(min=1e-5).to(dtype=weight_scales.dtype)
    o_scales = torch.repeat_interleave(
        group_scales.view(-1, params.head_dim), repeats=params.num_key_value_groups, dim=0
    )
    o_scales = o_scales.reshape(-1)
    v_scales = group_scales.reshape(-1)
    return o_scales, v_scales


class BaseScaleCalculator(ABC):
    """Abstract base class for scale calculators
    
    Defines the unified interface for scale computation in smooth quantization
    """
    
    @abstractmethod
    def compute_smooth_scale(
        self, 
        a_scale: torch.Tensor, 
        w_scale: torch.Tensor, 
        **kwargs
    ) -> torch.Tensor:
        pass
    
    @abstractmethod
    def compute_ov_scales(
        self,
        a_scale: torch.Tensor,
        w_scale: torch.Tensor,
        num_attention_heads: int,
        num_key_value_heads: int,
        **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        pass


class IterSmoothScaleCalculator(BaseScaleCalculator):
    """Scale calculator for IterSmooth
    
    Features:
    - Uses alpha to control smoothing, beta = 1 - alpha
    - Supports shift parameter
    - Uses mean method for MQGA
    """
    
    def __init__(self, alpha: float, scale_min: float = 1e-5):
        self.alpha = alpha
        self.scale_min = scale_min
    
    @torch.no_grad()
    def compute_smooth_scale(self, a_scale: torch.Tensor, w_scale: torch.Tensor, **kwargs) -> torch.Tensor:
        device = w_scale.device
        dtype = w_scale.dtype
        w_scale = w_scale.max(dim=0)[0].to(torch.float32).clamp(min=1e-5).to(dtype=dtype)
        scales = (a_scale.pow(self.alpha).to(device) / 
                  w_scale.pow(1 - self.alpha)).to(torch.float32).clamp(
                      min=self.scale_min).to(dtype=dtype)
        return scales
    
    @torch.no_grad()
    def compute_ov_scales(
        self,
        a_scale: torch.Tensor,
        w_scale: torch.Tensor,
        num_attention_heads: int,
        num_key_value_heads: int,
        **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        scales = self.compute_smooth_scale(a_scale, w_scale)
        shape_ratio, _ = prepare_mqga_parameters(num_attention_heads, num_key_value_heads)
        o_scales, v_scales = reduce_scales_for_mqga_mean(scales, shape_ratio, num_attention_heads)
        return o_scales, v_scales


class FlexSmoothScaleCalculator(BaseScaleCalculator):
    """Scale calculator for FlexSmoothQuant
    
    Features:
    - Uses both alpha and beta parameters
    - Supports two MQGA methods: mean and max
    - Searches for optimal parameters based on quantization error
    """
    
    def __init__(self, alpha: float, beta: float, group_method: str = 'max'):
        self.alpha = alpha
        self.beta = beta
        self.group_method = group_method
    
    @torch.no_grad()
    def compute_smooth_scale(self, a_scale: torch.Tensor, w_scale: torch.Tensor, **kwargs) -> torch.Tensor:
        scales = (
            (a_scale.pow(self.alpha) / w_scale.pow(self.beta))
            .to(torch.float32)
            .clamp(min=1e-5)
            .to(dtype=a_scale.dtype)
        )
        return scales
    
    @torch.no_grad()
    def compute_ov_scales(
        self,
        a_scale: torch.Tensor,
        w_scale: torch.Tensor,
        num_attention_heads: int,
        num_key_value_heads: int,
        **kwargs
        ) -> Tuple[torch.Tensor, torch.Tensor]:
        shape_ratio, _ = prepare_mqga_parameters(num_attention_heads, num_key_value_heads)
        head_dim = w_scale.shape[0] // num_attention_heads
        group_method = kwargs.get('group_method', self.group_method)
        
        if group_method == 'max':
            mqga_params = MQGAScaleParams(
                act_scales=a_scale,
                weight_scales=w_scale,
                best_alpha=self.alpha,
                best_beta=self.beta,
                num_key_value_groups=shape_ratio,
                head_dim=head_dim
            )
            o_scales, v_scales = reduce_scales_for_mqga_max(mqga_params)
            get_logger().debug(
                "group method: max, O scales shape: %s, V scales shape: %s",
                o_scales.shape, v_scales.shape
            )
        else:
            scales = self.compute_smooth_scale(a_scale, w_scale)
            o_scales, v_scales = reduce_scales_for_mqga_mean(scales, shape_ratio, num_attention_heads)
            get_logger().debug(
                "group method: mean, O scales shape: %s, V scales shape: %s",
                o_scales.shape, v_scales.shape
            )
        
        return o_scales, v_scales


class FlexAWQSSZScaleCalculator(BaseScaleCalculator):
    """Scale calculator for FlexAWQSSZ
    
    Features:
    - Searches for optimal alpha based on quantization config and actual quantization error
    - Beta is usually fixed at 0
    - Primarily uses max method for MQGA
    - Uses mean of activations instead of max
    """
    
    def __init__(self, alpha: float, beta: float = 0.0):
        self.alpha = alpha
        self.beta = beta
    
    @torch.no_grad()
    def compute_smooth_scale(self, a_scale: torch.Tensor, w_scale: torch.Tensor, **kwargs) -> torch.Tensor:
        scales = (
            (a_scale.pow(self.alpha) / w_scale.pow(self.beta))
            .to(torch.float32)
            .clamp(min=1e-5)
            .to(dtype=a_scale.dtype)
        )
        return scales
    
    @torch.no_grad()
    def compute_ov_scales(
        self,
        a_scale: torch.Tensor,
        w_scale: torch.Tensor,
        num_attention_heads: int,
        num_key_value_heads: int,
        **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        shape_ratio, _ = prepare_mqga_parameters(num_attention_heads, num_key_value_heads)
        head_dim = w_scale.shape[0] // num_attention_heads
        mqga_params = MQGAScaleParams(
            act_scales=a_scale,
            weight_scales=w_scale,
            best_alpha=self.alpha,
            best_beta=self.beta,
            num_key_value_groups=shape_ratio,
            head_dim=head_dim
        )
        o_scales, v_scales = reduce_scales_for_mqga_max(mqga_params)
        return o_scales, v_scales