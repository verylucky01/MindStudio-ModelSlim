## set_importance_evaluation_function

### 功能说明 
PruneTorch类方法，配置剪枝过程中用户自定义的重要性评估函数。用户未自定义时，默认是L1正则作为重要性。

### 函数原型
```python
set_importance_evaluation_function(importance_evaluation_function)
```

### 参数说明
| 参数名| 输入/返回值 | 含义 | 使用限制                    |
| ------ | ------ | ------ |-------------------------|
| importance_evaluation_function | 输入 | 自定义重要性评估函数，必须是可调用函数。| 必选。<br> 数据类型：函数。 |


### 调用示例
```python
from msmodelslim.pytorch.prune.prune_torch import PruneTorch
model = torchvision.models.vgg16(pretrained=False)
model.eval()
prune_torch= PruneTorch(model, torch.ones([1, 3, 224, 224]).type(torch.float32))
importance_evaluation_function = lambda chn_weight: torch.abs(chn_weight).mean().item()
prune_torch= prune_torch.set_importance_evaluation_function(importance_evaluation_function)
```