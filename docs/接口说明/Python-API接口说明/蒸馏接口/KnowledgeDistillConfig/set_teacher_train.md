## set_teacher_train

### 功能说明
KnowledgeDistillConfig类方法，通过调用此方法，将teacher模型进行训练。通常情况下，蒸馏只计算student模型的梯度并更新，teacher模型只用于推理，不需要调用此方法。但对于特殊的需求，可直接调用此方法将teacher进入梯度更新状态，无入参。

### 函数原型
```python
set_teacher_train()
```

### 调用示例
```python
from msmodelslim.common.knowledge_distill.knowledge_distill import KnowledgeDistillConfig
distill_config = KnowledgeDistillConfig()
distill_config.set_hard_label (0.5, 0)
distill_config.set_teacher_train()
```