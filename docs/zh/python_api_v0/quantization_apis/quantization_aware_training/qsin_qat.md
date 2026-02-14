## qsin_qat

### 功能说明 
模型量化接口，对用户提供的模型根据配置的量化参数进行量化。

### 函数原型
```python
qsin_qat(model, quant_config, quant_logger)
```

### 参数说明
| 参数名     | 输入/返回值 | 含义                                                      | 使用限制                   |
|---------| ------ |---------------------------------------------------------|------------------------|
| model   | 输入 | 待量化模型实例。| 必选。<br>数据类型：PyTorch模型。 |
|quant_config|输入|量化参数配置。| 必选。<br>数据类型：config。    |
|quant_logger|输入|量化输出日志。| 必选。<br>数据类型：log。       |



### 调用示例
```python
from msmodelslim.pytorch.quant.qat_tools import qsin_qat, QatConfig, get_logger
from torchvision.models import resnet50
import torch
model=resnet50()
quant_config = QatConfig()
quant_logger = get_logger()
model = qsin_qat(model, quant_config, quant_logger)
```