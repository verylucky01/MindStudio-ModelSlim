#  Copyright (c) 2025-2025 Huawei Technologies Co., Ltd.
#  #
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#  #
#  http://www.apache.org/licenses/LICENSE-2.0
#  #
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

"""
导入相关依赖
"""
import os
import torch
import torch.utils.data
from transformers import AutoTokenizer, AutoModelForCausalLM


torch.npu.set_compile_mode(jit_compile=False)
option = {}
option["NPU_FUZZY_COMPILE_BLACKLIST"] = "ReduceProd"
torch.npu.set_option(option)

SEQ_LEN_OUT = 32

"""
导入相关模型
""" 
LOAD_PATH = f"{os.environ['PROJECT_PATH']}/resource/llm_ptq/llama2_7b/"
tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=LOAD_PATH,
                                          trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path=LOAD_PATH,
                                             torch_dtype=torch.float32, trust_remote_code=True).cpu()

calib_list = ["Where is the capital of China?",
              "Please make a poem:",
              "I want to learn python, how should I learn it?",
              "Please help me write a job report on large model inference optimization:",
              "What are the most worth visiting scenic spots in China?"]
def get_calib_dataset(tokenizer, calib_list):
    calib_dataset = []
    for calib_data in calib_list:
        inputs = tokenizer([calib_data], return_tensors='pt').to('cpu')
        print(inputs)
        calib_dataset.append([inputs.data['input_ids'], inputs.data['attention_mask']])
    return calib_dataset
 
 
dataset_calib = get_calib_dataset(tokenizer, calib_list)
 
from msmodelslim.pytorch.llm_ptq.llm_ptq_tools.layer_select import LayerSelector
ls = LayerSelector(model, range_method='quantile')
ls.run(dataset_calib)
layers = ls.select_layers_by_threshold(1)
print("layer select by threshold:", layers)
layers = ls.select_layers_by_disable_level(2)
print("layer select by disable level:", layers)