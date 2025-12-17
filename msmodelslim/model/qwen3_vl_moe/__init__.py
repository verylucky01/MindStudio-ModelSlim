# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
"""
Qwen3-VL-MoE V1 Framework Adapter

This module provides v1 framework support for Qwen3-VL-MoE models with:
- Layer-wise loading and quantization
- Automatic MoE fusion layer conversion
- Multimodal calibration dataset handling
- Memory-efficient processing
"""

__all__ = [
    'Qwen3VLMoeV1ModelAdapter',
    'UnstackedQwen3VLMoeTextMLP',
    'UnstackedQwen3VLMoeSparseMoeBlock',
    'convert_qwen3_moe_to_linear',
]

from .model_adapter import Qwen3VLMoeModelAdapter
from .moe_utils import (
    UnstackedQwen3VLMoeTextMLP,
    UnstackedQwen3VLMoeSparseMoeBlock,
    convert_qwen3_moe_to_linear,
)