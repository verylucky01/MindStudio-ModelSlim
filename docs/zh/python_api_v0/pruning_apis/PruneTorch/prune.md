## prune

### 功能说明 
剪枝函数，配置剪枝过程中的各项参数，并返回剪枝信息，可在评估过程根据剪枝信息进行剪枝。

### 函数原型
```python
prune(reserved_ratio=0.75, un_prune_list=None)
```

### 参数说明
| 参数名| 输入/返回值 | 含义 | 使用限制                                                                                                                   |
| ------ | ------ | ------ |------------------------------------------------------------------------------------------------------------------------|
| reserved_ratio | 输入 | 剪枝参数量保留比例。| 可选。<br> 数据类型：Float。<br> 默认值为0.75，取值范围[0, 1]。                                                                           |
| un_prune_list | 输入 | 指定不剪枝的层，默认首尾不剪。| 可选。<br> 数据类型：list，元素必须是int或者string。<br>默认值为None。<br>若元素是int，说明是第几层不剪（只计算剪枝的算子Conv2d和Linear）。<br>若是string，表明是算子在网络中的名字。 |


### 调用示例
```python
from msmodelslim.pytorch.prune.prune_torch import PruneTorch
model = torchvision.models.vgg16(pretrained=False)
model.eval()
prune_torch = PruneTorch(model, torch.ones([1, 3, 224, 224]).type(torch.float32))
desc = prune_torch.prune(0.5)
```