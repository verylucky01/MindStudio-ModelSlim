## prune_model_weight

### 功能说明 
模型剪枝接口，根据原始的权重、较小参数加载的剪枝Transformer模型实例、剪枝配置传入接口，将原始的权重进行剪枝，并将剪枝后的权重载入较小参数的模型实例中。

### 函数原型
```python
prune_model_weight(model, config, weight_file_path)
```

### 参数说明
| 参数名| 输入/返回值 | 含义 | 使用限制                                |
| ------ | ------ | ----- |-------------------------------------|
| model | 输入 | 剪枝后模型实例。| 必选。<br> 数据类型：MindSpore模型或PyTorch模型。 |
| config | 输入 | 剪枝的配置。| 必选。<br> 数据类型：PruneConfig对象。。        |
| weight_file_path | 输入 | 剪枝前的原始模型权重文件所在路径及文件名。| 必选。<br> 数据类型：string <br> MindSpore模型的权重文件需为ckpt格式，PyTorch框架的权重文件需为pt/pth/pkl/bin格式。|


### 调用示例
```python
from msmodelslim.common.prune.transformer_prune.prune_model import PruneConfig
from msmodelslim.common.prune.transformer_prune.prune_model import prune_model_weight
# 定义配置类
prune_config = PruneConfig()
prune_config.set_steps(['prune_blocks', 'prune_bert_intra_block']). \
  add_blocks_params('uniter\.encoder\.encoder\.blocks\.(\d+)\.', {0: 1, 1: 3, 2: 5, 3: 7, 4: 9, 5: 11})
# 传入参数，对model进行剪枝
prune_model_weight(model, prune_config, weight_file_path = "xxx.ckpt")
```