# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.

import json
import os
import torch.utils.data
from transformers import AutoTokenizer, AutoModelForCausalLM
from modelslim.pytorch.llm_ptq.llm_ptq_tools import Calibrator, QuantConfig
from modelslim.pytorch.llm_ptq.anti_outlier import AntiOutlier, AntiOutlierConfig

# -------------------------- 获取脚本自身所在目录（不受执行目录影响） --------------------------
script_dir = os.path.dirname(os.path.abspath(__file__))

model_resource_path = os.environ.get("MODEL_RESOURCE_PATH")
if not model_resource_path:
    raise Exception("获取不到模型路径，请先检查环境变量 MODEL_RESOURCE_PATH")

fp16_path = os.path.join(model_resource_path, "Qwen2.5-7B-Instruct")
tokenizer = AutoTokenizer.from_pretrained(
    pretrained_model_name_or_path=fp16_path,
    trust_remote_code=True,
    local_files_only=True)
model = AutoModelForCausalLM.from_pretrained(
    pretrained_model_name_or_path=fp16_path,
    torch_dtype='auto',
    device_map='auto',
    trust_remote_code=True,
    local_files_only=True).eval()

disable_names = []
disable_names.append('lm_head')

model.eval()
w_sym = True
quant_config = QuantConfig(
    a_bit=16, w_bit=4, disable_names=disable_names, dev_type='npu', dev_id=model.device.index,
    w_sym=w_sym, mm_tensor=False, is_lowbit=True, open_outlier=False, group_size=64, w_method='HQQ')
calibrator = Calibrator(model, quant_config, calib_data=[], disable_level='L0')
calibrator.run()  # 执行PTQ量化校准

save_dir = os.path.join(os.path.join(script_dir, "output"), "llm_ptq_w4a16_pergroup_hqq")
calibrator.save(save_dir, save_type=["numpy", "safe_tensor"])
