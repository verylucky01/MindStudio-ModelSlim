# 导入相关依赖
import os
import json
import torch
import torch_npu   # 若需要在cpu上进行量化，可忽略此步骤
from transformers import AutoTokenizer, AutoModelForCausalLM

LOAD_PATH = f"{os.environ['PROJECT_PATH']}/resource/llm_ptq/llama2_7b/"
tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=LOAD_PATH,
                                          trust_remote_code=True, 
                                          local_files_only=True)
model = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path=LOAD_PATH,
                                             torch_dtype=torch.float16, 
                                             trust_remote_code=True, 
                                             local_files_only=True).npu()

# 量化配置，请根据实际情况修改
from msmodelslim.pytorch.llm_ptq.llm_ptq_tools import Calibrator, QuantConfig    # 导入量化配置接口
# 使用QuantConfig接口，配置量化参数，并返回量化配置实例
quant_config = QuantConfig(
    w_bit=8,    
    a_bit=16,         
    dev_id=model.device.index,
    dev_type='npu',   # 在cpu进行量化时，需配置参数dev_type='cpu'，并取消参数dev_id=model.device.index的配置
    act_method=3,
    pr=1.0,
    w_sym=True,
    mm_tensor=False,
    w_method='HQQ'
  )
#使用Calibrator接口，输入加载的原模型、量化配置和校准数据，定义校准
calibrator = Calibrator(model, quant_config, calib_data=[], disable_level='L0')  # Data Free场景下calib_data=[]
calibrator.run()     #使用run()执行量化
calibrator.save(f"{os.environ['PROJECT_PATH']}/output/llm_ptq_hqq", save_type=["numpy", "safe_tensor"])      #使用save()保存模型量化参数，请根据实际情况修改路径及保存的格式
print('Save quant weight success!')