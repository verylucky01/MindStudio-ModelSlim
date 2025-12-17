import os
import json
import re
import glob

import torch
import torch.utils.data
from transformers import AutoTokenizer, AutoModelForCausalLM

from modelslim.pytorch.llm_ptq.llm_ptq_tools import Calibrator, QuantConfig
from msmodelslim import logger as msmodelslim_logger

# for local path
fp16_path = f"{os.environ['PROJECT_PATH']}/resource/llm_ptq/llama2_7b/"  # 原始模型路径，其中的内容如下图
tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=fp16_path, 
                                          trust_remote_code=True, 
                                          local_files_only=True)
model = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path=fp16_path, 
                                             torch_dtype=torch.float32, 
                                             trust_remote_code=True, 
                                             local_files_only=True).cpu()

# 获取校准数据函数定义
def get_calib_dataset(tokenizer_instance, calib_list):
    calib_dataset = []
    for calib_data in calib_list:
        text = calib_data['inputs_pretokenized']
        inputs = tokenizer_instance([text], return_tensors='pt').to('cpu')
        msmodelslim_logger.info(inputs)
        calib_dataset.append([inputs.data['input_ids'], inputs.data['token_type_ids'], inputs.data['attention_mask']])
    return calib_dataset

entry = f"{os.environ['PROJECT_PATH']}/resource/CEval/val/Social_Science/teacher_qualification.jsonl"
dataset = []
with open(entry, encoding='utf-8') as file:
    for line in file:
        dataset.append(json.loads(line))

dataset_calib = get_calib_dataset(tokenizer, dataset[:1])

disable_names=[]
disable_names.append('lm_head')

model.eval()

# w_sym=True：对称量化，w_sym=False：非对称量化
w_sym = True
quant_config = QuantConfig(
    a_bit=16,
    w_bit=4,
    disable_names=disable_names,
    dev_type='cpu',
    w_sym=w_sym,
    mm_tensor=False,
    is_lowbit=True,
    open_outlier=False,
    w_method='GPTQ',
    group_size=128)

calibrator = Calibrator(model, quant_config, calib_data=dataset_calib, disable_level='L0')
calibrator.run()  # 执行PTQ量化校准
calibrator.save(f"{os.environ['PROJECT_PATH']}/output/llm_ptq_w4a16_pergroup_gptq", save_type=["numpy", "safe_tensor"])