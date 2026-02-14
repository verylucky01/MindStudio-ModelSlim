## __init__

### 功能说明 
PruneTorch类方法，对用户输入的模型进行类初始化。

### 函数原型
```python
__init__(network, inputs)
```

### 参数说明
| 参数名| 输入/返回值 | 含义 | 使用限制                    |
| ------ | ------ | ------ |-------------------------|
| network | 输入 | 待剪枝模型实例。| 必选。<br> 数据类型：PyTorch模型。 |
| inputs | 输入 | 模型的输入数据，用于解析模型。| 可选。<br> 数据类型：Tensor。    |


### 调用示例
```python
from msmodelslim.pytorch.prune.prune_torch import PruneTorch
model = torchvision.models.vgg16(pretrained=False)
model.eval()
prune_torch = PruneTorch(model, torch.ones([1, 3, 224, 224]).type(torch.float32))
```