import os
import json
import torch
import torch_npu
from transformers import AutoTokenizer, AutoModelForCausalLM
from modelslim.pytorch.llm_ptq.anti_outlier import AntiOutlierConfig, AntiOutlier
from modelslim.pytorch.llm_ptq.llm_ptq_tools import Calibrator, QuantConfig

fp16_path = f"{os.environ['PROJECT_PATH']}/resource/llm_ptq/llama2_7b/"
tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=fp16_path, 
                                          trust_remote_code=True, 
                                          local_files_only=True)
model = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path=fp16_path,
                                             trust_remote_code=True, 
                                             local_files_only=True).cpu()


def get_calib_dataset(fp16_tokenizer, calib_list, device="cpu"):
    calib_dataset = []
    for calib_data in calib_list:
        inputs = fp16_tokenizer(calib_data, return_tensors='pt')
        calib_dataset.append([
            inputs.data['input_ids'].to(device),
            inputs.data['attention_mask'].to(device)
        ])
    return calib_dataset


'''
# for 中文模型
calib_set = ["中国的首都在哪里？",
              "请做一首诗歌：",
              "我想要学习python，该怎么学习？",
              "请帮我写一篇关于大模型推理优化的任职报告：",
              "中国最值得去的几个景点"]
'''

calib_set = ["Where is the capital of China?",
             "Please make a poem:",
             "I want to learn python, how should I learn it?",
             "Please help me write a job report on large model inference optimization:",
             "What are the most worth visiting scenic spots in China?"]
dataset_calib = get_calib_dataset(tokenizer, calib_set)

disable_names = []
disable_names.append('lm_head')
quant_config = QuantConfig(
    w_bit=8,
    a_bit=8,
    disable_names=[],
    dev_type='cpu',  # dev_type="npu", dev_id=0  如果需要使用npu进行量化
    act_method=2,
    mm_tensor=False,
    do_smooth=True,
    is_lowbit=True,
    open_outlier=True,
)
calibrator = Calibrator(model, quant_config, calib_data=dataset_calib, disable_level='L0')
calibrator.run()
calibrator.save(f"{os.environ['PROJECT_PATH']}/output/llm_ptq_auto_optimize_w8a8", save_type=["numpy", "safe_tensor"])

print('Save quant weight success!')
