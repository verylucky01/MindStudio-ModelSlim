import os
from transformers import AutoTokenizer, AutoModelForCausalLM
from modelslim.pytorch.llm_ptq.llm_ptq_tools import Calibrator, QuantConfig

fp16_path = f"{os.environ['PROJECT_PATH']}/resource/llm_ptq/MOE_tiny/"
tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=fp16_path, 
                                          trust_remote_code=True, 
                                          local_files_only=True)
tokenizer.add_special_tokens({'pad_token': '[PAD]'})
tokenizer.pad_token_id = tokenizer.eos_token_id
model = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path=fp16_path,
                                             trust_remote_code=True, 
                                             local_files_only=True).cpu()

disable_names = []
disable_names.append('lm_head')
quant_config = QuantConfig(
    a_bit=8,
    w_bit=8,
    disable_names=disable_names,
    dev_type='cpu',
    pr=1.0,
    w_sym=True,
    mm_tensor=False,
    is_dynamic=True
)
calibrator = Calibrator(model, quant_config, calib_data=[], disable_level='L0')
calibrator.run()
calibrator.save(f"{os.environ['PROJECT_PATH']}/output/llm_ptq_w8a8_per_token_moe", save_type=["numpy", "safe_tensor"])

print('Save quant weight success!')