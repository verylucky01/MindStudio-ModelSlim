# 混合校准数据集使用方法说明

## 简介

混合校准集接口，通过CalibrationData类混合指定的数据集，支持用户自定义数据集

## 使用前准备

前提条件：config文件，用于配置基础数据集的路径，名称包括 boolq、ceval_5_shot、gsm8k、mmlu
    <br>数据集下载链接
    ```
    https://huggingface.co/datasets/ceval/ceval-exam
    https://huggingface.co/datasets/google/boolq
    https://huggingface.co/datasets/cais/mmlu
    https://huggingface.co/datasets/openai/gsm8k
    ```
## 功能介绍

### 接口说明
请参考 [CalibrationData](../../../python_api/foundation_model_compression_apis/foundation_model_quantization_apis/CalibrationData.md)

操作步骤：
1. 如需自定义数据集，创建自定义数据集处理类，继承自DatasetProcessorBase类，并重写process_data()和verify_positive_prompt()方法
2. 实例化CalibrationData，如需正样本混合校准集，需要实例化tokenizer和model，并作为参数传入CalibrationData；否则设置为None。如需保存需要设置保存路径
3. 如有自定义数据集，通过add_customized_dataset_processor()接口传入自定义数据集名称和处理类的实例
4. 设置样本数量，通过set_sample_size()接口
5. 设置batch_size，通过set_batch_size()接口
6. 设置随机种子，通过set_shuffle_seed()接口
7. 调用process接口运行，生成混合校准集

### config文件示例
- 第一层为dict，key为"configurations"，value为一个list，包含多个数据集信息
- 每个数据集为一个dict，key为"dataset_name"和"dataset_path"，用来配置数据集的名称和路径
```json
{"configurations": 
    [
        {
          "dataset_name": "boolq",
          "dataset_path": "./boolq/dev.jsonl"
        },
        {
          "dataset_name": "ceval_5_shot",
          "dataset_path": "./ceval_5_shot/"
        },
        {
          "dataset_name": "gsm8k",
          "dataset_path": "./gsm8k/GSM8K.jsonl"
        },
        {
          "dataset_name": "mmlu",
          "dataset_path": "./mmlu/"
        }
    ]  
}
```

### 调用示例
请注意`trust_remote_code`为`True`时可能执行浮点模型权重中代码文件，请确保浮点模型来源安全可靠。
```python
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig 

from msmodelslim.pytorch.llm_ptq.mix_calibration.calib_select import CalibrationData
from msmodelslim.pytorch.llm_ptq.mix_calibration.dataset_processor_base import DatasetProcessorBase # 用户自定义数据集时需要引入

# 继承自DatasetProcessorBase，并重写抽象方法
class CustomizedProcessor(DatasetProcessorBase):
    def __init__(self, dataset_path, tokenizer=None, model=None):
        super().__init__(dataset_path, tokenizer, model)
        self.ori_prompts = []
        self.ori_answers = []
    
    def process_data(self, indexs):
        """用于获取一组样本，输出为[{"prompt": prompt1, "ans": ans1},{"prompt": prompt2, "ans": ans2}]"""
        prmpts_anses = []
        for idx in indexs:
            prmpts_anses.append({"prompt": self.ori_prompts[idx], "ans": self.ori_answers[idx]})
        return prmpts_anses
    
    def verify_positive_prompt(self, prompts, labels):
        """用于验证一组prompts中的正样本，labels为对应标签，输出为[{"prompt": prompt1, "ans": ans1},{"prompt": prompt2, "ans": ans2}]"""
        prpt_ans = []
        with torch.no_grad():
            inputs = self.tokenizer(prompts, padding=True, return_tensors="pt").to(self.model.device)
            outputs = self.model.generate(**inputs, do_sample=False, max_new_tokens=20)
            
            answers = []
            for idx in range(len(outputs)):
                output = outputs.tolist()[idx][len(inputs["input_ids"][idx]):]
                response = self.tokenizer.decode(output)
                answers.append(response)
            answers = [answer.lstrip()[0] if answer.lstrip() else "-1" for answer in answers]

            for ans, label, prmpt in zip(answers, labels, prompts):
                if ans == label:
                    prpt_ans.append({"prompt": prmpt, "ans": ans})

        return prpt_ans

MODEL_PATH = "./model"
CONFIG_PATH = "./mix_config.json"
SAVE_PATH = "./mix_dataset.json"

config = AutoConfig.from_pretrained(MODEL_PATH, trust_remote_code=True, local_files_only=True)
tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=MODEL_PATH,
                                          trust_remote_code=True, 
                                          local_files_only=True)
model = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path=MODEL_PATH,
                                             trust_remote_code=True,
                                             config=config,
                                             torch_dtype='auto',
                                             device_map='auto', 
                                             local_files_only=True)

# 基础支持的校准集包括boolq、ceval_5_shot、gsm8k、mmlu。 customized_dataset_name为用户自定义数据集名称
# 当sample_size中设置了非config.json中配置的数据集且非用户自定义数据集名称时，会报错“Dataset {dataset_name} has no handler”
# 当sample_size为空时，返回空结果
sample_size = {"boolq": 4, "ceval_5_shot": 3, "gsm8k": 3, "mmlu": 2, "customized_dataset_name": 3}

# 用户自定义数据集
customized_dataset_path = "./customized_dataset"
customized_processor = CustomizedProcessor(customized_dataset_path, tokenizer=tokenizer, model=model)

calib_select = CalibrationData(config_path=CONFIG_PATH, save_path=SAVE_PATH, tokenizer=tokenizer, model=model)  # 若不需要正样本，tokenizer和model均置为None
calib_select.add_customized_dataset_processor("customized_dataset_name", customized_processor)     # 该调用需在设置采样数量之前
calib_select.set_sample_size(sample_size)
calib_select.set_batch_size(4)  # 该调用仅用于设置获取正样本时的batch，不对输出产生影响， 输入为int类型
calib_select.set_shuffle_seed(1)

mixed_dataset = calib_select.process()
print(mixed_dataset)
```

### 混合校准集解析使用示例
通过get_anti_dataset()方法，以混合校准集生成的mixed_dataset为输入，输出可以应用于离群值抑制模块``AntiOutlier(model, calib_data=mixed_dataset, cfg=anti_config)``中``calib_data``的输入 <br>
通过get_calib_dataset()方法，以混合校准集生成的mixed_dataset为输入，输出可以应用于量化模块``Calibrator(model, quant_config, calib_data=mixed_dataset, disable_level='L0')``中``calib_data``的输入
```python
import torch
import torch.nn.functional as F

def get_anti_dataset(tokenizer, mixed_dataset, device="npu"):
    """用于离群值抑制的校准集"""
    anti_data = []
    for prpt_ans in mixed_dataset:
        calib_dataset = []
        calib_list = [prpt_ans["prompt"]]
        max_len = 0
        for calib_data in calib_list:
            inputs = tokenizer(calib_data, return_tensors='pt')
            calib_dataset.append(inputs.data['input_ids'].to(device))
            max_len = max(max_len, inputs.data['input_ids'].size(1)) 
        for i in range(len(calib_dataset)):
            calib_dataset[i] = F.pad(calib_dataset[i], (0, max_len - calib_dataset[i].size(1)), value=0)
        anti_data.append(torch.cat(calib_dataset))
    
    anti_dataset = []
    for data in anti_data:
        anti_dataset.append([data])
    
    return anti_dataset

def get_calib_dataset(tokenizer, mixed_dataset, device='npu'):
    """用于量化的校准集"""
    dataset_calib = []
    for prpt_ans in mixed_dataset:
        calib_list = [prpt_ans["prompt"]]
        calib_dataset = []
        for calib_data in calib_list:
            inputs = tokenizer(calib_data, return_tensors='pt').to(device)
            calib_dataset.append([inputs.data['input_ids']])
        dataset_calib += calib_dataset

    return dataset_calib
```
