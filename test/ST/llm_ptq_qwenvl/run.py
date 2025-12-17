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
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from PIL import Image
import requests
import os 

from msmodelslim.pytorch.llm_ptq.llm_ptq_tools import Calibrator, QuantConfig
from msmodelslim.pytorch.llm_ptq.anti_outlier import AntiOutlier, AntiOutlierConfig


torch.npu.set_compile_mode(jit_compile=False)
option = {}
option["NPU_FUZZY_COMPILE_BLACKLIST"] = "ReduceProd"
torch.npu.set_option(option)

"""
导入相关模型
"""
LOAD_PATH = f"{os.environ['PROJECT_PATH']}/resource/llm_ptq/qwen-vl/"
tokenizer = AutoTokenizer.from_pretrained(LOAD_PATH,
                                            trust_remote_code=True)

model = AutoModelForCausalLM.from_pretrained(
    LOAD_PATH, device_map='auto', trust_remote_code=True , torch_dtype=torch.float16).eval()
 
 
images_list = os.listdir(f"{os.environ['PROJECT_PATH']}/resource/llm_ptq/coco2/")
image = Image.open(os.path.join(f"{os.environ['PROJECT_PATH']}/resource/llm_ptq/coco2/", images_list[0]))
prompt = "USER: <image>\nDescribe this image in detail. ASSISTANT:"
calib_data = []
for i in images_list[:1]:
    imagePath = os.path.join(f"{os.environ['PROJECT_PATH']}/resource/llm_ptq/coco2/", i)
    query = tokenizer.from_list_format([
        {'image': imagePath}, # Either a local path or an url
        {'text': prompt},
    ])
    inputs = tokenizer(query, return_tensors='pt')
    inputs = inputs.to(model.device)
    calib_data.append([inputs['input_ids'], None, inputs['attention_mask']])
 
 
disable_names = []
disable_names.append("transformer.visual.attn_pool.attn.out_proj")

"""
对于linear算子中的激活值如果有表示范围过大，或者"尖刺"的异常值过多，
需要使用anti outlier功能，使用方法如下
"""
anti_config = AntiOutlierConfig(
    a_bit=8,
    w_bit=8,
    anti_method="m2",
    dev_type="npu",
    dev_id=model.device.index,
    )
anti_outlier = AntiOutlier(model, calib_data=calib_data , cfg=anti_config)
anti_outlier.process()
 
 
quant_config = QuantConfig(
    w_bit=8,
    a_bit=8,
    disable_names=disable_names,
    dev_type='npu',
    dev_id=model.device.index,
    act_method=2,
    mm_tensor=False
)
 
calibrator = Calibrator(model, quant_config, calib_data=calib_data, disable_level='L0')
calibrator.run()
calibrator.save(f"{os.environ['PROJECT_PATH']}/output/llm_ptq_qwenvl")