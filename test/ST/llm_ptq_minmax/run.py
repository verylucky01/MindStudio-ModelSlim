import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from modelslim.pytorch.llm_ptq.llm_ptq_tools import Calibrator, QuantConfig

# for local path
LOAD_PATH = f"{os.environ['PROJECT_PATH']}/resource/llm_ptq/llama2_7b/"
model = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path=LOAD_PATH, 
                                             torch_dtype=torch.float32,
                                             trust_remote_code=True, 
                                             local_files_only=True).cpu()

disable_names=[]
disable_names.append('lm_head')

# w_sym=True：对称量化，w_sym=False：非对称量化
w_sym = False
quant_config = QuantConfig(a_bit=16, w_bit=8, disable_names=disable_names, dev_type='cpu', act_method=3, pr=1.0,
                           w_sym=w_sym, mm_tensor=False, w_method='MinMax')
calibrator = Calibrator(model, quant_config, calib_data=[], disable_level='L0')
calibrator.run()  # 执行PTQ量化校准
calibrator.save(f"{os.environ['PROJECT_PATH']}/output/llm_ptq_minmax", save_type=["numpy", "safe_tensor"])