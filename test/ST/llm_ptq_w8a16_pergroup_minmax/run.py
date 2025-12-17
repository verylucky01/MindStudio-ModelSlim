import json
import os
import torch.utils.data
from transformers import AutoTokenizer, AutoModelForCausalLM
from modelslim.pytorch.llm_ptq.llm_ptq_tools import Calibrator, QuantConfig
from modelslim.pytorch.llm_ptq.anti_outlier import AntiOutlier, AntiOutlierConfig

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
disable_names = []
disable_names.append('lm_head')

model.eval()
w_sym = True
quant_config = QuantConfig(a_bit=16, w_bit=8, disable_names=disable_names, dev_type='cpu', w_sym=w_sym,
                           mm_tensor=False, is_lowbit=True, open_outlier=False, group_size=64, w_method='MinMax')
calibrator = Calibrator(model, quant_config, calib_data=[], disable_level='L0')
calibrator.run()  # 执行PTQ量化校准

calibrator.save(f"{os.environ['PROJECT_PATH']}/output/llm_ptq_w8a16_pergroup_minmax", save_type=["numpy", "safe_tensor"])