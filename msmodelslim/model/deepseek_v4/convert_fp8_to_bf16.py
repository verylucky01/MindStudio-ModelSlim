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
from functools import lru_cache
from typing import Dict

import torch
import torch.distributed as dist
from safetensors import safe_open
from torch import nn
from tqdm import tqdm

from msmodelslim.utils.logging import get_logger
from msmodelslim.utils.security import get_valid_read_path, MAX_READ_FILE_SIZE_32G, json_safe_load

WEIGHT_SCALE_INV = '.scale'
HF_HOOK = '_hf_hook'
MTP_PREFIX = 'mtp.0.'


def decode_fp8(
    weight: torch.Tensor,
    scale: torch.Tensor,
) -> torch.Tensor:
    weight = weight.unflatten(0, (-1, 128)).unflatten(-1, (-1, 128)).float() * scale[:, None, :, None].float()
    return weight.flatten(2, 3).flatten(0, 1).bfloat16()


def decode_fp4(
    packed_fp4_data: torch.Tensor,
    block_scales: torch.Tensor,
) -> torch.Tensor:
    lut = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], device=packed_fp4_data.device, dtype=torch.float32)

    # 解包
    uint8 = packed_fp4_data.view(torch.uint8)
    low = uint8 & 0x0F
    high = (uint8 >> 4) & 0x0F
    indices = torch.stack([low, high], dim=-1).flatten(-2)

    # 解码
    sign = 1.0 - 2.0 * ((indices >> 3) & 1).float()
    abs_idx = indices & 0x07
    values = sign * lut[abs_idx.long()]

    # 应用缩放 （block_size = 2048 / 128 = 16）
    scales_expanded = block_scales.to(torch.float32).repeat_interleave(32, dim=-1)  # (4096, 2048)
    result = (values * scales_expanded).to(torch.bfloat16)

    return result


@lru_cache(maxsize=1)
def get_inv_weight_map(model_path: str):
    model_index_path = os.path.join(model_path, "model.safetensors.index.json")
    model_index = json_safe_load(model_index_path)
    weight_map = model_index['weight_map']
    weight_map = {k.replace(WEIGHT_SCALE_INV, ''): v for k, v in weight_map.items() if WEIGHT_SCALE_INV in k}
    return weight_map


def get_inv_tensor(tensor_name, fp8_path, weight_map):
    file_name = weight_map[tensor_name]
    file_path = os.path.join(fp8_path, file_name)
    file_path = get_valid_read_path(file_path, 'safetensors', size_max=MAX_READ_FILE_SIZE_32G)
    
    # 自动检测设备类型
    if dist.is_initialized():
        if hasattr(torch, 'npu') and torch.npu.is_available():
            current_device_idx = torch.npu.current_device()
            device = f"npu:{current_device_idx}"
        elif hasattr(torch, 'cuda') and torch.cuda.is_available():
            current_device_idx = torch.cuda.current_device()
            device = f"cuda:{current_device_idx}"
        else:
            device = 'cpu'
    else:
        device = 'cpu'
    
    with safe_open(file_path, framework='pt', device=device) as f:
        return f.get_tensor(tensor_name + WEIGHT_SCALE_INV)


def get_real_name(name, layer_prefix=''):
    return f'{layer_prefix}.{name}' if layer_prefix else name


def auto_dequant_state_dict(layer_prefix: str, state_dict: dict, model_path: str, mtp_layer_prefix: str = "model.layers.43."):
    weight_map = get_inv_weight_map(model_path)

    if not weight_map:
        return

    try:
        sub_weight_map = {}
        for sub_name in state_dict:
            sub_key = get_real_name(sub_name, layer_prefix).replace(mtp_layer_prefix, MTP_PREFIX).replace('.weight', '')
            if sub_key in weight_map:
                sub_weight_map[sub_key] = weight_map[sub_key]
        dequant_state_dict(layer_prefix, state_dict, model_path, weight_map=sub_weight_map, mtp_layer_prefix=mtp_layer_prefix)
    except KeyError:
        get_logger().warning(f'Safetensors files not match index.json, please check whether model is of bf16.')
        get_logger().warning(f'Skip fp8 to bf16.')


@torch.no_grad()
def dequant_state_dict(layer_prefix: str,
                              module: dict,
                              model_path: str,
                              weight_map: Dict[str, str],
                              mtp_layer_prefix: str):
    with tqdm(total=len(weight_map), desc='fp8 to bf16') as bar:
        for ori_name, weight in module.items():
            real_name = get_real_name(ori_name, layer_prefix)
            sub_name = real_name.replace(mtp_layer_prefix, MTP_PREFIX).replace('.weight', '')
            if sub_name not in weight_map:
                continue

            weight_dequant = decode_fp8 if str(weight.dtype) == 'torch.float8_e4m3fn' else decode_fp4

            scale = get_inv_tensor(sub_name, model_path, weight_map)
            module[ori_name] = weight_dequant(weight, scale.to(weight.device)).detach()
            weight.to('meta')
            bar.update(1)
