#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""
-------------------------------------------------------------------------
This file is part of the MindStudio project.
Copyright (c) 2026 Huawei Technologies Co.,Ltd.

MindStudio is licensed under Mulan PSL v2.
You can use this software according to the terms and conditions of the Mulan PSL v2.
You may obtain a copy of Mulan PSL v2 at:

         http://license.coscl.org.cn/MulanPSL2

THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
See the Mulan PSL v2 for more details.
-------------------------------------------------------------------------
"""

import os

import torch
import torch.nn as nn
from safetensors import safe_open
from safetensors.torch import load_file

from ascend_utils.common.security.path import get_valid_read_path
from msmodelslim.utils.logging import get_logger
from msmodelslim.utils.security import json_safe_load
from msmodelslim.utils.exception import UnexpectedError

from .convert_fp8_to_bf16 import auto_dequant_state_dict

MAX_READ_FILE_SIZE_16G = 17179869184  # 16G, 16 * 1024 * 1024 * 1024


def remove_zero_and_shift(matrix):
    n, m = matrix.shape

    # Step 1: 找到每行第一个 0 的位置（即要删除的位置）
    # 如果某行没有 0，则默认保留所有元素（但根据题意，应该每行都有一个 0）
    zero_pos = (matrix == 0).int().argmax(dim=1)  # [n,]

    # Step 2: 构造掩码，标记要保留的元素（排除每行的第一个 0）
    # 生成一个 [n, m] 的坐标矩阵，标记每列是否等于 zero_pos
    col_indices = torch.arange(m, device=matrix.device).expand(n, -1)  # [n, m]
    mask = (col_indices != zero_pos.unsqueeze(1))  # [n, m]

    # Step 3: 用掩码筛选元素（自动展平，需要重新调整形状）
    filtered = matrix[mask].view(n, m - 1)  # [n, m-1]

    # Step 4: 在最后一列补 0
    result = torch.cat([filtered, torch.zeros(n, 1, device=matrix.device)], dim=1)  # [n, m]

    return result.to(matrix)


class SharedHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.norm = NewModelRMSNorm(
            config.dim, eps=1e-06
        )
        self.head = nn.Linear(config.dim, config.vocab_size, bias=False)

    def forward(self, hidden_states):
        normalized_states = self.norm(hidden_states)
        logits = self.head(normalized_states)
        return logits


class NewModelRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        NewModelRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


class NewModelEmbedding(nn.Module):
    def __init__(self, vocab_size, hidden_size):
        super().__init__()
        self.tok_emb = nn.Embedding(
            vocab_size, hidden_size
        )

    def forward(self, input_ids):
        return self.tok_emb(input_ids)


class MTPLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.enorm = NewModelRMSNorm(
            config.dim, eps=1e-06
        )
        self.hnorm = NewModelRMSNorm(
            config.dim, eps=1e-06
        )
        self.e_proj = nn.Linear(
            config.dim,
            config.dim,
            bias=False
        )
        self.h_proj = nn.Linear(
            config.dim,
            config.dim,
            bias=False
        )

        self.head = nn.Linear(config.dim, config.vocab_size, bias=False)
        self.norm = NewModelRMSNorm(
            config.dim, eps=1e-06
        )

        self.emb = NewModelEmbedding(
            config.vocab_size, config.dim
        )

        hc_mult = config.hc_mult
        hc_dim = hc_mult * config.dim
        self.hc_head_fn = nn.Parameter(torch.empty(hc_mult, hc_dim))
        self.hc_head_base = nn.Parameter(torch.empty(hc_mult))
        self.hc_head_scale = nn.Parameter(torch.empty(1))


def get_shared_weight(model_path, key):
    # since some weights in mtp_layer like 'emb.tok_emb.weight'/'head.weight' are shared with the main model part,
    # we need to get the weight according to the index.json
    index_json_path = os.path.join(model_path, "model.safetensors.index.json")
    index_data = json_safe_load(index_json_path)

    if key not in index_data["weight_map"]:
        raise UnexpectedError(f"{key} is not found. Please check whether {key} is shared with the main model part.")

    weight_path = os.path.join(model_path, index_data["weight_map"][key])
    with safe_open(weight_path, framework='pt', device='cpu') as f:
        weight = f.get_tensor(key)   
    
    return weight


def get_mtp_layer(config, model_path, layer_prefix, mtp_layer_prefix="model.layers.43."):
    get_logger().debug('Start to load mtp')
    mtp_layer = MTPLayer(config)
    mtp_safetensor = os.path.join(model_path, "model-00046-of-00046.safetensors")
    mtp_safetensor = get_valid_read_path(
        mtp_safetensor,
        size_max=MAX_READ_FILE_SIZE_16G,
        is_dir=False,
        check_user_stat=True
    )
    mtp_weight = load_file(mtp_safetensor, device="cpu")
    new_state_dict = {}
    # get shared weights for mtp_layer from main model part
    new_state_dict['emb.tok_emb.weight'] = get_shared_weight(model_path, 'embed.weight')
    new_state_dict['head.weight'] = get_shared_weight(model_path, 'head.weight')
    # remove prefix for mtp_layer
    for key, value in mtp_weight.items():
        new_key = key.replace('mtp.0.', '')
        if new_key in mtp_layer.state_dict().keys():
            new_state_dict[new_key] = value
    auto_dequant_state_dict(layer_prefix, new_state_dict, model_path, mtp_layer_prefix)
    get_logger().debug('Success to convert mtp_layer from fp8 to bf16')
    mtp_layer.load_state_dict(new_state_dict)
    get_logger().debug('Success to load mtp')
    return mtp_layer


def wrap_mtp_decoder(mtp_decoder: nn.Module, mtp_layer: nn.Module):
    get_logger().debug('Start to wrap mtp')
    mtp_decoder.enorm = mtp_layer.enorm
    mtp_decoder.hnorm = mtp_layer.hnorm
    # mtp_decoder.eh_proj = mtp_layer.eh_proj
    mtp_decoder.e_proj = mtp_layer.e_proj
    mtp_decoder.h_proj = mtp_layer.h_proj
    # mtp_decoder.shared_head = mtp_layer.head
    mtp_decoder.head = mtp_layer.head
    mtp_decoder.norm = mtp_layer.norm
    # mtp_decoder.embed_tokens = mtp_layer.embed_tokens
    mtp_decoder.emb = mtp_layer.emb
    mtp_decoder.hc_head_fn = mtp_layer.hc_head_fn
    mtp_decoder.hc_head_base = mtp_layer.hc_head_base
    mtp_decoder.hc_head_scale = mtp_layer.hc_head_scale
    get_logger().debug('Success to wrap mtp')
