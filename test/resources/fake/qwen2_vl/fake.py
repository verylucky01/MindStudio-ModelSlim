# Copyright (c) Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.
import os

from transformers.models.qwen2_vl.configuration_qwen2_vl import Qwen2VLConfig
from transformers.models.qwen2_vl.modeling_qwen2_vl import Qwen2VLVisionBlock, Qwen2VLDecoderLayer


class FakeQwenCreator:
    """
    用于生成一个随机的、非常小的Qwen2VLVisionBlock，用于验证工具中某些流程的正确性
    """

    config = Qwen2VLConfig.from_pretrained(os.path.join(os.path.dirname(__file__), "config.json"))
    vision_config = config.vision_config

    @classmethod
    def get_block(cls):
        """
        获取一个随机的、非常小的Qwen2VLVisionBlock，用于验证工具中某些流程的正确性
        """
        block = Qwen2VLVisionBlock(config=cls.vision_config, attn_implementation="eager")
        return block

    @classmethod
    def get_decoder_layer(cls):
        """
        获取一个随机的、非常小的Qwen2VLDecoderLayer，用于验证工具中某些流程的正确性
        """
        layer = Qwen2VLDecoderLayer(config=cls.config, layer_idx=0)
        return layer
