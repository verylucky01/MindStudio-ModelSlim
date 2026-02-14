## add_blocks_params

### 功能说明 
PruneConfig类方法，根据自定义参数配置模型剪枝的block，若set_steps选择的步骤包含“prune_blocks”，则需要调用该方法。

### 函数原型
```python
add_blocks_params(pattern, layer_id_map)
```

### 参数说明
| 参数名| 输入/返回值 | 含义 | 使用限制                                      |
| ------ | ------ | ------ |-------------------------------------------|
| pattern | 输入 | 待剪枝网络layer名称的正则表达式。| 必选。<br> 数据类型：string。<br> 例如取值为"bert\\.encoder\\.layer\\.(\d+)"时，表示选取网络中以bert.encoder.layer开头，且后续为数字的网络layer。<br>说明:在使用复杂的正则表达式时，用户需保证正则表达式的安全性，规避ReDoS攻击的风险，否则会引起程序执行缓慢。|
| layer_id_map | 输入 | 待剪枝网络layer的前后id匹配关系。| 必选。<br> 数据类型：dict，key和value的数据类型均为int。<br> 例如，取值为{0: 0, 1: 2, 2: 4}时表示将bert.encoder.layer.0的权重保留至bert.encoder.layer.0，bert.encoder.layer.2的权重保留至bert.encoder.layer.1，bert.encoder.layer.4的权重保留至bert.encoder.layer.2，即预训练权重中bert.encoder.layer.x共有5层，而输入的模型中bert.encoder.layer.x只有3层，通过layer_id_map在剪枝时将权重保留到指定的位置。|


### 调用示例
```python
from msmodelslim.common.prune.transformer_prune.prune_model import PruneConfig
prune_config = PruneConfig()
prune_config.set_steps(['prune_blocks']). \
  add_blocks_params('uniter\.encoder\.encoder\.blocks\.(\d+)\.', {0: 1, 1: 3, 2: 5, 3: 7, 4: 9, 5: 11})
```