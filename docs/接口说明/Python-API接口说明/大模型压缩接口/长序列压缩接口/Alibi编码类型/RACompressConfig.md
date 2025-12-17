## RACompressConfig

### 功能说明 
长序列压缩时，配置压缩过程中的参数。

### 函数原型
```python
RACompressConfig(theta=0.00001, alpha=100)
```

### 参数说明
| 参数名| 输入/返回值 | 含义 | 使用限制 |
| ------ | ------ | ------ | ------ |
| theta | 输入 | attention score贡献度，保证校准后模型推理精度。| 可选。<br>数据类型：float。<br>默认为0.00001，可选范围为[0.00001, 0.001]。 |
| alpha | 输入 | 校准偏置，用于保证适用广度，控制窗口大小。| 可选。<br>数据类型：int。<br>默认为100，可选范围为[0, 10000]。 |


### 调用示例
```python
from msmodelslim.pytorch.ra_compression import RACompressConfig
config = RACompressConfig(theta=0.00001, alpha=100)
```