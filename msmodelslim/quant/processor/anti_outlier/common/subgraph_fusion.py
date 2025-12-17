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
from typing import Dict, Optional

import torch
from torch import nn

from msmodelslim.core.QAL.qtypes import (
    Subgraph,
    OVSubgraph,
    UpDownSubgraph,
    LinearLinearSubgraph,
    NormLinearSubgraph
)


@torch.no_grad()
def apply_smooth_scale_shift(
    layer: nn.Module, 
    scales: torch.Tensor, 
    shift: Optional[torch.Tensor] = None
) -> None:
    device = layer.weight.device
    dtype = layer.weight.dtype
    layer.weight.mul_(scales)
    if shift is not None:
        shift = shift.to(device).to(dtype)
        if layer.bias is None:
            # If no bias exists, create new bias parameter
            bias_shape = (layer.weight.shape[0],)
            layer.bias = nn.Parameter(
                torch.zeros(bias_shape, device=device, dtype=dtype, requires_grad=False)
            )
        layer.bias.add_(shift)


class SubgraphFusionStrategy(ABC): 
    @abstractmethod
    def apply_fusion(
        self, 
        subgraph: Subgraph, 
        scales: Dict[str, torch.Tensor],
        shifts: Optional[Dict[str, torch.Tensor]] = None
    ) -> None:
        pass


class OVSubgraphFusion(SubgraphFusionStrategy):
    def apply_fusion(
        self, 
        subgraph: OVSubgraph, 
        scales: Dict[str, torch.Tensor],
        shifts: Optional[Dict[str, torch.Tensor]] = None
    ) -> None:
        o_scales = scales['o_scales']
        v_scales = scales['v_scales']
        
        o_shift = shifts.get('o_shift') if shifts else None
        v_shift = shifts.get('v_shift') if shifts else None
        apply_smooth_scale_shift(subgraph.o_proj, o_scales.view(1, -1), o_shift)
        apply_smooth_scale_shift(
            subgraph.v_proj, 1.0 / v_scales.view(-1, 1), v_shift
        )


class UpDownSubgraphFusion(SubgraphFusionStrategy):
    def apply_fusion(
        self, 
        subgraph: UpDownSubgraph, 
        scales: Dict[str, torch.Tensor],
        shifts: Optional[Dict[str, torch.Tensor]] = None
    ) -> None:
        scales_tensor = scales['scales']
        
        down_shift = shifts.get('down_shift') if shifts else None
        up_shift = shifts.get('up_shift') if shifts else None
        apply_smooth_scale_shift(
            subgraph.down_proj, scales_tensor.view(1, -1), down_shift
        )
        apply_smooth_scale_shift(
            subgraph.up_proj, 1.0 / scales_tensor.view(-1, 1), up_shift
        )


class LinearLinearSubgraphFusion(SubgraphFusionStrategy):
    def apply_fusion(
        self, 
        subgraph: LinearLinearSubgraph, 
        scales: Dict[str, torch.Tensor],
        shifts: Optional[Dict[str, torch.Tensor]] = None
    ) -> None:
        scales_tensor = scales['scales']
        
        linear2_shift = shifts.get('linear2_shift') if shifts else None
        linear1_shift = shifts.get('linear1_shift') if shifts else None
        apply_smooth_scale_shift(
            subgraph.linear2, scales_tensor.view(1, -1), linear2_shift
        )
        apply_smooth_scale_shift(
            subgraph.linear1, 1.0 / scales_tensor.view(-1, 1), linear1_shift
        )


class NormLinearSubgraphFusion(SubgraphFusionStrategy):
    def apply_fusion(
        self, 
        subgraph: NormLinearSubgraph, 
        scales: Dict[str, torch.Tensor],
        shifts: Optional[Dict[str, torch.Tensor]] = None
    ) -> None:
        scales_tensor = scales['scales']
        linear_shifts = shifts.get('linear_shifts', []) if shifts else []
        norm_shift = shifts.get('norm_shift') if shifts else None
        for idx, fc in enumerate(subgraph.linears):
            linear_shift = linear_shifts[idx] if idx < len(linear_shifts) else None
            apply_smooth_scale_shift(fc, scales_tensor.view(1, -1), linear_shift)
        apply_smooth_scale_shift(
            subgraph.norm, (1.0 / scales_tensor).squeeze(), norm_shift
        )


class SubgraphFusionFactory:
    
    _fusers = {
        OVSubgraph: OVSubgraphFusion(),
        UpDownSubgraph: UpDownSubgraphFusion(),
        LinearLinearSubgraph: LinearLinearSubgraphFusion(),
        NormLinearSubgraph: NormLinearSubgraphFusion(),
    }
    
    @classmethod
    def get_fuser(cls, subgraph: Subgraph) -> Optional[SubgraphFusionStrategy]:
        for registered_type, fuser in cls._fusers.items():
            if isinstance(subgraph, registered_type):
                return fuser
        return None
    
    @classmethod
    def apply_fusion_to_subgraph(
        cls,
        subgraph: Subgraph,
        scales: Dict[str, torch.Tensor],
        shifts: Optional[Dict[str, torch.Tensor]] = None
    ) -> None:
        fuser = cls.get_fuser(subgraph)
        if fuser is None:
            raise ValueError(f"Unsupported subgraph type: {type(subgraph).__name__}")
        fuser.apply_fusion(subgraph, scales, shifts)