# -*- coding: UTF-8 -*-
# Copyright 2025 The ZhipuAI Inc. team and HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# Copyright (c) 2026 Huawei Technologies Co.,Ltd.

import gc

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.activations import ACT2FN
from transformers.models.glm4v_moe.modeling_glm4v_moe import (
    Glm4vMoeTextMoE,
    Glm4vMoeTextNaiveMoe,
)

__all__ = [
    "UnstackedGlm4vTextExpertMLP",
    "UnstackedGlm4vTextTopkRouter",
    "UnstackedGlm4vMoeTextMoE",
]


class UnstackedGlm4vTextExpertMLP(nn.Module):
    """Single expert MLP using nn.Linear, matching the safetensors key layout."""

    def __init__(self, hidden_size: int, intermediate_size: int, hidden_act: str, dtype=None):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False, dtype=dtype)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False, dtype=dtype)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False, dtype=dtype)
        self.act_fn = ACT2FN[hidden_act]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class UnstackedGlm4vTextTopkRouter(nn.Module):
    """
    Router compatible with Glm4vMoeTextTopkRouter, but stores e_score_correction_bias
    as nn.Parameter so the saver can persist it.

    modelslim_v1 saver only writes named_parameters(...), not buffers. If we keep
    e_score_correction_bias as register_buffer like transformers does, it participates
    in forward but disappears in exported weights. Promoting it to Parameter preserves
    the original key path:

        mlp.gate.weight
        mlp.gate.e_score_correction_bias
    """

    def __init__(self, config, original_gate: nn.Module):
        super().__init__()
        self.config = config
        self.top_k = config.num_experts_per_tok
        self.n_routed_experts = config.n_routed_experts
        self.routed_scaling_factor = config.routed_scaling_factor
        self.n_group = config.n_group
        self.topk_group = config.topk_group
        self.norm_topk_prob = config.norm_topk_prob

        self.weight = nn.Parameter(torch.empty_like(original_gate.weight))
        self.e_score_correction_bias = nn.Parameter(original_gate.e_score_correction_bias.detach().clone())

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = hidden_states.view(-1, self.config.hidden_size)
        return F.linear(hidden_states.type(torch.float32), self.weight.type(torch.float32))


class UnstackedGlm4vMoeTextMoE(nn.Module):
    """
    Drop-in replacement for Glm4vMoeTextMoE, but with per-expert nn.Linear modules:

        mlp.experts.0.gate_proj.weight
        mlp.experts.0.up_proj.weight
        mlp.experts.0.down_proj.weight

    路由与前向逻辑直接复刻自 transformers.Glm4vMoeTextMoE / Glm4vMoeTextNaiveMoe：
        - route_tokens_to_experts: 与 Glm4vMoeTextMoE.route_tokens_to_experts 一致；
        - _dispatch_to_experts:    与 Glm4vMoeTextNaiveMoe.forward 一致，只是改为调用
                                   每个 expert 的 Linear MLP。
    """

    def __init__(self, config, original_moe: Glm4vMoeTextMoE):
        super().__init__()

        self.config = config
        self.n_routed_experts = config.n_routed_experts
        self.n_group = config.n_group
        self.topk_group = config.topk_group
        self.norm_topk_prob = config.norm_topk_prob
        self.routed_scaling_factor = config.routed_scaling_factor
        self.top_k = config.num_experts_per_tok

        # shared_experts 直接复用；gate 改为 Parameter 版本，确保 e_score_correction_bias
        # 能被 saver 识别并写回导出权重。
        self.gate = UnstackedGlm4vTextTopkRouter(config, original_moe.gate)
        self.shared_experts = original_moe.shared_experts

        # experts: 直接是 ModuleList[UnstackedGlm4vTextExpertMLP]
        self.num_experts = config.num_local_experts
        dtype = next(original_moe.experts.parameters()).dtype
        self.experts = nn.ModuleList(
            [
                UnstackedGlm4vTextExpertMLP(
                    config.hidden_size,
                    config.moe_intermediate_size,
                    config.hidden_act,
                    dtype=dtype,
                )
                for _ in range(self.num_experts)
            ]
        )

        # 释放原 NaiveMoe 中的 3D fused 权重，避免无用内存占用
        for attr in ("gate_up_proj", "down_proj"):
            if hasattr(original_moe.experts, attr):
                delattr(original_moe.experts, attr)
        gc.collect()

    def route_tokens_to_experts(self, router_logits: torch.Tensor):
        """
        与 Glm4vMoeTextMoE.route_tokens_to_experts 保持一致。
        """
        router_logits = router_logits.sigmoid()
        router_logits_for_choice = router_logits + self.gate.e_score_correction_bias
        group_scores = (
            router_logits_for_choice.view(-1, self.n_group, self.n_routed_experts // self.n_group)
            .topk(2, dim=-1)[0]
            .sum(dim=-1)
        )
        group_idx = torch.topk(group_scores, k=self.topk_group, dim=-1, sorted=False)[1]
        group_mask = torch.zeros_like(group_scores)
        group_mask.scatter_(1, group_idx, 1)
        score_mask = (
            group_mask.unsqueeze(-1)
            .expand(-1, self.n_group, self.n_routed_experts // self.n_group)
            .reshape(-1, self.n_routed_experts)
        )
        scores_for_choice = router_logits_for_choice.masked_fill(~score_mask.bool(), 0.0)
        topk_indices = torch.topk(scores_for_choice, k=self.top_k, dim=-1, sorted=False)[1]
        topk_weights = router_logits.gather(1, topk_indices)
        if self.norm_topk_prob:
            denominator = topk_weights.sum(dim=-1, keepdim=True) + 1e-20
            topk_weights = topk_weights / denominator
        topk_weights = topk_weights * self.routed_scaling_factor
        return topk_indices, topk_weights

    def _dispatch_to_experts(
        self,
        hidden_states: torch.Tensor,
        top_k_index: torch.Tensor,
        top_k_weights: torch.Tensor,
    ) -> torch.Tensor:
        """
        与 Glm4vMoeTextNaiveMoe.forward 等价，只是改为调用 per-expert Linear MLP。
        """
        final_hidden_states = torch.zeros_like(hidden_states)
        with torch.no_grad():
            expert_mask = torch.nn.functional.one_hot(top_k_index, num_classes=self.num_experts)
            expert_mask = expert_mask.permute(2, 1, 0)
            expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()

        for expert_idx_tensor in expert_hit:
            expert_idx = expert_idx_tensor[0].item()
            if expert_idx >= self.num_experts:
                continue
            top_k_pos, token_idx = torch.where(expert_mask[expert_idx])
            current_hidden_states = self.experts[expert_idx](hidden_states[token_idx])
            current_hidden_states = current_hidden_states * top_k_weights[token_idx, top_k_pos, None]
            final_hidden_states.index_add_(0, token_idx, current_hidden_states.to(final_hidden_states.dtype))

        return final_hidden_states

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        与 Glm4vMoeTextMoE.forward 保持相同的输入/输出接口。
        """
        residuals = hidden_states
        orig_shape = hidden_states.shape
        router_logits = self.gate(hidden_states)
        topk_indices, topk_weights = self.route_tokens_to_experts(router_logits)
        hidden_states = hidden_states.view(-1, hidden_states.shape[-1])
        hidden_states = self._dispatch_to_experts(hidden_states, topk_indices, topk_weights).view(*orig_shape)
        hidden_states = hidden_states + self.shared_experts(residuals)
        return hidden_states
