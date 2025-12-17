## get_compress_heads

### 功能说明 
执行RARopeCompressor后，可通过调用get_compress_heads()函数在指定路径下生成.pt文件。

### 函数原型
```python
RARopeCompressor.get_compress_heads(save_path)
```

### 参数说明
| 参数名| 输入/返回值 | 含义 | 使用限制 |
| ------ | ------ | ------ | ------ |
| save_path | 输入 | 长序列压缩时，Head压缩头参数文件保存的路径。| 必选。<br>数据类型：String。 |


### 调用示例
```python
import torch
from msmodelslim.pytorch.ra_compression import RARopeCompressConfig, RARopeCompressor
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch_npu
torch.npu.set_compile_mode(jit_compile=False)
 
config = RARopeCompressConfig(induction_head_ratio=0.14, echo_head_ratio=0.01)
 
save_path = "./win.pt" 
model_path = "/home/wgw/Meta-Llama-3.1-70B-Instruct/"
 
model = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path=model_path,
        local_files_only=True,
        torch_dtype=torch.bfloat16, 
        device_map="auto"
    ).eval()
 
tokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path=model_path,
        local_files_only=True,
        pad_token='<|extra_0|>',
        eos_token='<|endoftext|>',
        padding_side='left'
    ) 
 
ra = RARopeCompressor(model, tokenizer, config) 
ra.get_compress_heads(save_path)
```