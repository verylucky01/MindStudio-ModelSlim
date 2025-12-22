## prune_by_desc

### 功能说明 
根据已有的剪枝信息，在推理时对模型进行剪枝。

### 函数原型
```python
prune_by_desc(desc)
```

### 参数说明
| 参数名| 输入/返回值 | 含义 | 使用限制                     |
| ------ | ------ | ----- |--------------------------|
| desc | 输入 | 剪枝信息。| 必选。<br> 数据类型：prune或analysis接口返回的desc信息。 |


### 调用示例
```python
from msmodelslim.pytorch.prune.prune_torch import PruneTorch
model = torchvision.models.vgg16(pretrained=False)
model.eval()
prune_torch = PruneTorch(model, torch.ones([1, 3, 224, 224]).type(torch.float32))
desc = prune_torch.prune()
prune_torch.prune_by_desc(desc)
```