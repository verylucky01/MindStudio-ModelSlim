#  Copyright (c) Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.
import os

from modelslim.pytorch.ra_compression import RACompressConfig, RACompressor
from transformers import AutoTokenizer, AutoModelForCausalLM

config = RACompressConfig(theta=0.00001, alpha=100)
input_model_path = f"{os.environ['PROJECT_PATH']}/resource/llm_ptq/baichuan2-13b/"
output_model_path = f"{os.environ['PROJECT_PATH']}/output/ra_compression_baichuan/win.pt"
tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=input_model_path, 
                                          trust_remote_code=True, 
                                          local_files_only=True)
model = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path=input_model_path,
                                             trust_remote_code=True, 
                                             local_files_only=True).float().cpu()
ra = RACompressor(model, config)
ra.get_alibi_windows(output_model_path)
print('success!')