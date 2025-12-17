# Copyright Huawei Technologies Co., Ltd. 2025. All rights reserved.

from enum import Enum
from typing import Callable, Dict, Tuple

import torch


class AttentionType(Enum):
    """注意力机制类型"""
    MHA = "mha"  # Multi-Head Attention
    MQA = "mqa"  # Multi-Query Attention
    GQA = "gqa"  # Group-Query Attention
    MLA = "mla"  # Multi-Head Latent Attention


class ForwardFactory:
    """用于管理不同模型类型和注意力类型/处理器类型的forward函数适配器的工厂类"""

    _forward_adapters: Dict[Tuple[str, str], Callable] = {}


    @classmethod
    def register(cls, model_type: str, attn_or_processor_type: str):
        """装饰器，用于注册forward适配器"""
        def decorator(func):
            key = (model_type, attn_or_processor_type)
            cls._forward_adapters[key] = func
            return func
        return decorator


    @classmethod
    def get_forward_adapter(cls, model_type: str, attn_or_processor_type: str):
        """获取指定模型类型和注意力类型的forward适配器"""
        key = (model_type, attn_or_processor_type)
        if key not in cls._forward_adapters:
            raise ValueError(
                f"Unsupported combination: model_type={model_type}, "
                f"attn_or_processor_type={attn_or_processor_type}"
            )
        return cls._forward_adapters[key]


    @classmethod
    def detect_attention_type(cls, module: torch.nn.Module) -> str:
        """检测模块的注意力类型"""
        if not hasattr(module, "num_key_value_heads"):
            return AttentionType.MHA.value

        if module.num_key_value_heads == module.num_attention_heads:
            return AttentionType.MHA.value
        elif module.num_key_value_heads == 1:
            return AttentionType.MQA.value
        elif module.num_key_value_heads < module.num_attention_heads:
            return AttentionType.GQA.value

        return AttentionType.MHA.value
