## set_steps

### 功能说明 
PruneConfig类方法，根据自定义参数配置模型剪枝的步骤。

### 函数原型
```python
set_steps(steps)
```

### 参数说明
| 参数名| 输入/返回值 | 含义 | 使用限制                                      |
| ------ | ------ | ------ |-------------------------------------------|
| steps | 输入 | 权重剪枝的步骤。| 必选。<br> 数据类型：list。<br> 取值如下：<br> 1. “prune_bert_intra_block”：根据输入的模型进行预训练权重裁剪。裁剪预训练权重与模型中同名但是shape不同的权重，从而使预训练权重的shape与模型一致，可单独指定。<br>数据类型：string。<br> 2. “prune_blocks”：根据add_blocks_params()的参数进行预训练权重裁剪。将指定id的layer的权重保留到另一个layer，可单独指定。指定该步骤时，须同时配置add_blocks_params方可生效。<br> 数据类型：string。|


### 调用示例
```python
from msmodelslim.common.prune.transformer_prune.prune_model import PruneConfig
prune_config = PruneConfig()
prune_config.set_steps([ 'prune_bert_intra_block'])
```