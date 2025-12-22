## set_node_reserved_ratio

### 功能说明 
PruneTorch类方法，配置模型剪枝过程中算子节点保留的参数比例。

### 函数原型
```python
set_node_reserved_ratio(node_reserved_ratio)
```

### 参数说明
| 参数名| 输入/返回值 | 含义 | 使用限制                     |
| ------ | ------ | ------ |--------------------------|
| node_reserved_ratio | 输入 | 剪枝过程中最多裁剪掉的算子节点比例。| 必选。<br> 数据类型：Float。<br> 取值范围0-1。 |


### 调用示例
```python
from msmodelslim.pytorch.prune.prune_torch import PruneTorch
model = torchvision.models.vgg16(pretrained=False)
model.eval()
prune_torch= PruneTorch(model, torch.ones([1, 3, 224, 224]).type(torch.float32))
prune_torch= prune_torch.set_node_reserved_ratio(0.5)
```