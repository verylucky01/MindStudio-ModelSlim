# -*- coding: utf-8 -*-
# Copyright Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.
import os
import stat

import torch
import pytest
from transformers import AutoConfig, AutoModelForCausalLM, PreTrainedTokenizerBase, PreTrainedTokenizerBase

from ascend_utils.common.security import file_safe_write, safe_delete_path_if_exists
from msmodelslim.pytorch.ra_compression import RARopeCompressConfig, RARopeCompressor


torch.manual_seed(2024)


@pytest.fixture(scope="module", autouse=True)
def model_config():
    model_config_path = "./config.json"
    config = """{
      "architectures": [
        "LlamaForCausalLM"
      ],
      "bos_token_id": 1,
      "eos_token_id": 2,
      "hidden_act": "silu",
      "hidden_size": 1024,
      "initializer_range": 0.02,
      "intermediate_size": 4096,
      "max_position_embeddings": 4096,
      "model_type": "llama",
      "num_attention_heads": 1,
      "num_hidden_layers": 1,
      "num_key_value_heads": 1,
      "pad_token_id": 0,
      "pretraining_tp": 1,
      "rms_norm_eps": 1e-05,
      "rope_scaling": null,
      "tie_word_embeddings": false,
      "torch_dtype": "float16",
      "transformers_version": "4.31.0.dev0",
      "use_cache": true,
      "vocab_size": 32000
    }"""
    file_safe_write(config, model_config_path)
    os.chmod(model_config_path, stat.S_IRUSR | stat.S_IWUSR)
    yield
    safe_delete_path_if_exists(model_config_path)


class FakeTokenizer(PreTrainedTokenizerBase):
    def __call__(self, text, return_tensors='pt'):
        return {
            "input_ids": torch.arange(32)[None],
            "attention_mask": torch.zeros([1, 32])
        }


@pytest.mark.skip()
def test_ra_compression_given_model_then_pass():
    output_model_path = "./win.pt"

    config = AutoConfig.from_pretrained("./", local_files_only=True)
    model = AutoModelForCausalLM.from_config(config)
    tokenizer = FakeTokenizer()

    config = RARopeCompressConfig(induction_head_ratio=0.9, echo_head_ratio=0.01)
    ra = RARopeCompressor(model, tokenizer, config)
    ra.get_compress_heads(output_model_path)

    head_dict = torch.load(output_model_path)
    assert head_dict["prefix_matching"][0] == [0]
    safe_delete_path_if_exists(output_model_path)