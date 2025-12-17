# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
"""
Multimodal VLM V1 Quantization Service

A unified quantization service for multimodal vision-language models with:
- Automatic MoE fusion layer conversion
- Layer-wise loading and processing
- Multi-modal calibration dataset support
- Compatible with msmodelslim quant command

Supported models:
- Qwen3-VL-MoE
- Other multimodal VLM models (extensible)
"""

__all__ = [
    'MultimodalVLMModelslimV1QuantService',
    'MultimodalVLMModelslimV1QuantConfig',
]

from .quant_service import MultimodalVLMModelslimV1QuantService
from .quant_config import MultimodalVLMModelslimV1QuantConfig