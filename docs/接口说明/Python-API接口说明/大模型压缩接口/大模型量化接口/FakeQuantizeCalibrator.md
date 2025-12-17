## FakeQuantizeCalibrator

### 功能说明
基于量化权重将浮点模型转换为伪量化模型。

### 函数原型
```python
FakeQuantizeCalibrator(model, dev_id, dev_type, description, safetensor)
```

### 参数说明
| 参数名| 输入/返回值 | 含义 | 使用限制 |
| ------ | ------ | ------ | ------ |
| model | 输入 | 模型。| 必选。<br>数据类型：nn.Module。 |
| dev_id | 输入 | Device ID。| 必选。<br>数据类型：int。<br>仅在“dev_type”配置为“npu”时生效。“dev_id”指定的Device ID优先级高于环境变量配置的Device ID。 |
| dev_type | 输入 | Device类型。| 必选。<br>数据类型：str。<br>可选值：['cpu', 'npu']，默认为'cpu'。 |
| description | 输入 | 量化后生成的json描述文件。| 必选。<br>数据类型：dict。 |
| safetensor | 输入 | 量化后生成的safetensors格式的权重文件。| 必选。<br>数据类型：dict。 |

### 调用示例
根据实际需求，在QuantConfig初始化中完成所有参数的配置。
```python
import torch
import json
from safetensors.torch import load_file
from transformers import AutoTokenizer, AutoModelForCausalLM
from msmodelslim.pytorch.llm_ptq.llm_ptq_tools import FakeQuantizeCalibrator
if __name__ == '__main__':
    fp16_path = './chatglm2_6b/'  # 文件路径
    model = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path=fp16_path, 
                                                 local_files_only=True, 
                                                 torch_dtype=torch.float32).cpu()
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=fp16_path, local_files_only=True,)
    safetensor_dic = load_file('./quant_model_weight_w8a16.safetensors')  # 使用load_file()函数读取safetensor格式文件并将其解析为字典
    with open('./quant_model_description_w8a16.json', 'r', encoding='utf-8') as file:
        description_dic = json.load(file)  # 使用json.load()函数读取文件并将其解析为字典
    fakecalibrator = FakeQuantizeCalibrator(model, None, "cpu", description_dic, safetensor_dic)
    model = fakecalibrator.model
```
