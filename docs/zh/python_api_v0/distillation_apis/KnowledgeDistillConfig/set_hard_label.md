## set_hard_label

### 功能说明
KnowledgeDistillConfig类方法，配置蒸馏时student模型的loss权重、loss的index，必须调用该方法。

### 函数原型
```python
set_hard_label(weight, index)
```

### 参数说明
| 参数名| 输入/返回值 | 含义 | 使用限制 |
| ------ | ------ | ------ | ------ |
| weight | 输入 | hard loss 的权重。| 必选。<br>数据类型：float。<br>建议取值0-1之间。 |
| index | 输入 | student模型output的索引值(index)。当student模型有多个output时，由index决定哪个output来计算loss。<br>仅MindSpore模型需要配置，通常为0。| 必选。<br>数据类型：int。 |


### 调用示例
```python
from msmodelslim.common.knowledge_distill.knowledge_distill import KnowledgeDistillConfig
distill_config = KnowledgeDistillConfig()
distill_config.set_hard_label (0.5, 0)
```