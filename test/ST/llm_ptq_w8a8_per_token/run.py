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

anti_config = AntiOutlierConfig(anti_method="m1", dev_type="cpu")
anti_outlier = AntiOutlier(model, calib_data=dataset_calib, cfg=anti_config)
anti_outlier.process()

disable_names = []
llama_layers = 1
disable_idx_lst = list(range(llama_layers))[:2]
for layer_index in disable_idx_lst:
    down_proj_name = "model.layers.{}.mlp.down_proj".format(layer_index)
    disable_names.append(down_proj_name)
disable_names.append('lm_head')
quant_config = QuantConfig(
    a_bit=8,
    w_bit=8,
    disable_names=disable_names,
    dev_type='cpu',
    act_method=1,
    pr=1.0,
    w_sym=True,
    mm_tensor=False,
    is_dynamic=True
)
calibrator = Calibrator(model, quant_config, calib_data=dataset_calib, disable_level='L0')
calibrator.run()
calibrator.save(f"{os.environ['PROJECT_PATH']}/output/llm_ptq_w8a8_per_token", save_type=["numpy", "safe_tensor"])

print('Save quant weight success!')