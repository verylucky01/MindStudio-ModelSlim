## RACompressor

### 功能说明 
压缩参数配置类，通过RACompressor可获得长序列压缩所需的权重文件（.pt文件）。

### 函数原型
```python
RACompressor(model, cfg)
```

### 参数说明
| 参数名| 输入/返回值 | 含义 | 使用限制 |
| ------ | ------ | ------ | ------ |
| model | 输入 | 当前支持的模型。<br>说明：不支持使用npu方式进行加载。| 必选。<br>模型类型：PyTorch模型。|
| cfg | 输入 | RACompressConfig的配置。| 必选。<br>数据类型：int。<br>配置类：RACompressConfig。 |


### 调用示例
```python
from msmodelslim.pytorch.ra_compression import RACompressConfig, RACompressor
model = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path="baichuan2-13b/float_path/", 
                                             local_files_only=True).float().cpu()  # 需根据模型的实际路径配置
config = RACompressConfig(theta=0.00001, alpha=100)
ra = RACompressor(model,config) 
```