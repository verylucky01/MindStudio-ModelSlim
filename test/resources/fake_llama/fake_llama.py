# Copyright (c) Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.
import os

from transformers import AutoTokenizer
from transformers.models.llama import LlamaConfig, LlamaForCausalLM


def get_fake_llama_model_and_tokenizer():
    """
    获取一个随机的、非常小的Llama模型以及其所对应的tokenizer，用于验证工具中某些数值算法的正确性
    """
    # 使用绝对路径确保无论从哪个目录运行测试都能正确找到配置文件
    current_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(current_dir, "config.json")
    config = LlamaConfig.from_json_file(config_path)
    tokenizer = AutoTokenizer.from_pretrained(current_dir)
    return LlamaForCausalLM(config), tokenizer