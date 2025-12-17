import os
import json
import torch
from safetensors.torch import load_file
from transformers import AutoTokenizer, AutoModelForCausalLM
from msmodelslim.pytorch.llm_ptq.llm_ptq_tools import FakeQuantizeCalibrator

# for local path
LOAD_PATH = f"{os.environ['PROJECT_PATH']}/resource/llm_ptq/llama2_7b/"
model = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path=LOAD_PATH, 
                                             torch_dtype=torch.float32,
                                             trust_remote_code=True, 
                                             local_files_only=True).cpu()
tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=LOAD_PATH, 
                                          trust_remote_code=True, 
                                          local_files_only=True)
# 使用load_file()函数读取safetensor格式文件并将其解析为字典
safetensor_dic = load_file(f"{os.environ['PROJECT_PATH']}/resource/llm_ptq_minmax/quant_model_weight_w8a16.safetensors")
# 使用json.load()函数读取文件并将其解析为字典
with open(f"{os.environ['PROJECT_PATH']}/resource/llm_ptq_minmax/quant_model_description_w8a16.json", 'r',
          encoding='utf-8') as file:
    description_dic = json.load(file)
fakecalibrator = FakeQuantizeCalibrator(model, None, "cpu", description_dic, safetensor_dic)
model = fakecalibrator.model
print('fake quant weight success!')