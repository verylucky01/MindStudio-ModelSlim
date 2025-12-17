# Copyright Huawei Technologies Co., Ltd. 2025. All rights reserved.

import json
import os

from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig

from msmodelslim.utils.exception import InvalidModelError, InvalidDatasetError
from msmodelslim.utils.exception_decorator import exception_handler
from msmodelslim.utils.security import get_valid_read_path

MAX_KEY_LENGTH = 256
MAX_JSON_LENGTH = 4096


class SafeGenerator:
    def __init__(self):
        pass

    @staticmethod
    def get_config_from_pretrained(model_path, **kwargs):
        model_path = get_valid_read_path(model_path, is_dir=True, check_user_stat=True)
        with exception_handler(f'Get config from pretrained failed in {model_path}.',
                               err_cls=Exception,
                               ms_err_cls=InvalidModelError,
                               action=f"Please ensure config files all exist and are valid. "
                                      f"Otherwise, the transformers version is not compatible with the model."
                                      f"Before using msModelSlim, please make sure the model load and infer properly.",
                               ):
            config = AutoConfig.from_pretrained(model_path, local_files_only=True, **kwargs)
            return config

    @staticmethod
    def get_model_from_pretrained(model_path, **kwargs):
        model_path = get_valid_read_path(model_path, is_dir=True, check_user_stat=True)
        with exception_handler(f'Get model from pretrained failed in {model_path}.',
                               err_cls=Exception,
                               ms_err_cls=InvalidModelError,
                               action=f"Please ensure the model weights files all exist and are valid. "
                                      f"Otherwise, the transformers version is not compatible with the model."
                                      f"Before using msModelSlim, please make sure the model load and infer properly.",
                               ):
            model = AutoModelForCausalLM.from_pretrained(model_path, local_files_only=True, **kwargs)
            return model

    @staticmethod
    def get_tokenizer_from_pretrained(model_path, **kwargs):
        model_path = get_valid_read_path(model_path, is_dir=True, check_user_stat=True)
        with exception_handler(f'Get tokenizer from pretrained failed in {model_path}.',
                               err_cls=Exception,
                               ms_err_cls=InvalidModelError,
                               action=f"Please ensure the tokenizer files all exist and are valid. "
                                      f"Otherwise, the transformers version is not compatible with the model."
                                      f"Before using msModelSlim, please make sure the model load and infer properly.",
                               ):
            tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True, **kwargs)
            return tokenizer

    @staticmethod
    def load_jsonl(dataset_path, key_name='inputs_pretokenized'):
        dataset = []
        if dataset_path == "humaneval_x.jsonl":
            key_name = 'prompt'
        with os.fdopen(os.open(dataset_path, os.O_RDONLY, 0o600),
                       'r', encoding='utf-8') as file:
            lines = file.readlines()
            for line in lines:
                data = json.loads(line)
                text = data.get(key_name, line)
                dataset.append(text)
        return dataset