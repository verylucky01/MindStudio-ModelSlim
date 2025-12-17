# Copyright Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.
import os
import json
import shutil
import stat
import torch
import torch.utils.data
from transformers import AutoTokenizer, AutoModelForCausalLM
from modelslim.pytorch.llm_ptq.anti_outlier import AntiOutlier, AntiOutlierConfig
from modelslim.pytorch.llm_ptq.llm_ptq_tools import Calibrator, QuantConfig

IN_MODEL_PATH = f"{os.environ['PROJECT_PATH']}/resource/llm_ptq/llama2_7b/"
OUT_MODEL_PATH = f"{os.environ['PROJECT_PATH']}/output/llm_ptq_bf16"
NUM_LAYERS = 2  #
ANTI_METHOD = "m1"


def get_calib_dataset(_tokenizer, _calib_list):
    calib_dataset = []
    for calib_data in _calib_list:
        inputs = _tokenizer([calib_data], return_tensors='pt')
        print(inputs)
        calib_dataset.append([inputs.data['input_ids'].cpu(), None, inputs.data['attention_mask'].cpu()])
    return calib_dataset


tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=IN_MODEL_PATH, 
                                          local_files_only=True)
model = AutoModelForCausalLM.from_pretrained(
    pretrained_model_name_or_path=IN_MODEL_PATH,
    torch_dtype=torch.bfloat16, 
    use_safetensors=True, 
    local_files_only=True).cpu()
print(f"loading success!")

calib_list = [
    "Where is the capital of China?",
]

dataset_calib = get_calib_dataset(tokenizer, calib_list)

print("quantization start...")
disabled_names = []
disabled_layers = [i for i in range(0, NUM_LAYERS)]
for i in disabled_layers:
    disabled_names.append(f"model.layers.{i}.mlp.down_proj")

quant_config = QuantConfig(a_bit=8, w_bit=8, disable_names=disabled_names, dev_type='cpu',
                           act_method=3, mm_tensor=False)

calibrator = Calibrator(model, quant_config, calib_data=dataset_calib, disable_level='L0')

calibrator.run()
print("quantization success!")

calibrator.save(OUT_MODEL_PATH, save_type=["numpy", "safe_tensor"])

print(f"saved successfully")