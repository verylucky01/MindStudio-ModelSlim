# Copyright (c) Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.

import os

from transformers.models.clip.modeling_clip import CLIPEncoderLayer
from transformers.models.llama.modeling_llama import LlamaDecoderLayer, LlamaRotaryEmbedding
from transformers.models.llava.configuration_llava import LlavaConfig


class FakeLlavaCreator:
    """
    用于生成一个随机的、非常小的LLaVA模型组件，用于验证工具中某些流程的正确性
    """

    config = LlavaConfig.from_pretrained(os.path.join(os.path.dirname(__file__), "config.json"))
    vision_config = config.vision_config
    text_config = config.text_config

    @classmethod
    def get_vision_block(cls):
        """
        获取一个随机的、非常小的LlavaClipVision，用于验证工具中某些流程的正确性
        """
        block = CLIPEncoderLayer(config=cls.vision_config)
        return block

    @classmethod
    def get_decoder_layer(cls):
        """
        获取一个随机的、非常小的LlavaDecoderLayer，用于验证工具中某些流程的正确性
        """
        rotary_emb = LlamaRotaryEmbedding(
            config=cls.text_config
        )
        layer = LlamaDecoderLayer(config=cls.text_config, layer_idx=0)
        layer.self_attn.rotary_emb = rotary_emb
        layer.self_attn.num_heads = cls.text_config.num_attention_heads
        layer.self_attn.head_dim = cls.text_config.hidden_size // cls.text_config.num_attention_heads
        layer.self_attn.num_key_value_heads = cls.text_config.num_attention_heads
        layer.self_attn.num_key_value_groups = cls.text_config.num_attention_heads
        return layer
