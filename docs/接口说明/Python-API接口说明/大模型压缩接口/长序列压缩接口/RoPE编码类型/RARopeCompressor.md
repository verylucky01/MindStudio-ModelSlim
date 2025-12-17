## RARopeCompressor

### 功能说明 
压缩参数配置类，通过RARopeCompressor可获得长序列压缩所需的权重文件。

### 函数原型
```python
RARopeCompressor(model, tokenizer, cfg)
```

### 参数说明
| 参数名| 输入/返回值 | 含义 | 使用限制 |
| ------ | ------ | ------ | ------ |
| model | 输入 | 当前支持的模型。| 必选。<br>模型类型：PyTorch模型。 |
| tokenizer | 输入 | 用于加载预训练模型的tokenizer。| 必选。<br>类型：AutoTokenizer。 |
| cfg | 输入 | RARopeCompressConfig的配置。| 必选。<br>配置类：RARopeCompressConfig。 |


### 调用示例
```python
from msmodelslim.pytorch.ra_compression import RARopeCompressConfig, RARopeCompressor
config = RARopeCompressConfig(induction_head_ratio=0.14, echo_head_ratio=0.01)
ra = RARopeCompressor(model, tokenizer, config) 
```