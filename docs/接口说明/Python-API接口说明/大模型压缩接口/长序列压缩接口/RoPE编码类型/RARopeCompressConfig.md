## RARopeCompressConfig

### 功能说明 
RoPE编码的模型进行长序列压缩时，配置Induction Head和Echo Head的保留比例，保留的部分不需压缩，仅需压缩未保留部分。

### 函数原型
```python
RARopeCompressConfig(induction_head_ratio=0.14, echo_head_ratio=0.01)
```

### 参数说明
| 参数名| 输入/返回值 | 含义 | 使用限制 |
| ------ | ------ | ------ | ------ |
| induction_head_ratio | 输入 | 控制Induction Head的保留比例。<br>说明：Induction Head：在处理文本序列时，用于关注并预测输入序列中与当前token（词元）相同的下一个token。| 可选。<br>数据类型：float。<br>默认为0.14，可选范围为[0,1]。 |
| echo_head_ratio | 输入 | 控制Echo Head的保留比例。<br>说明：Echo Head：在处理文本序列时，用于关注前文中出现的与当前token相同的token。| 可选。<br>数据类型：float。<br>默认为0.01，可选范围为[0,1]。 |


### 调用示例
```python
from msmodelslim.pytorch.ra_compression import RARopeCompressConfig
config = RARopeCompressConfig(induction_head_ratio=0.14, echo_head_ratio=0.01)
```