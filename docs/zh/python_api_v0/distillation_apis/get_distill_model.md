## get_distill_model

### 功能说明
模型蒸馏接口，将用户提供教师模型、学生模型根据蒸馏配置进行组合，返回一个DistillDualModels实例，用户对DistillDualModels 实例进行训练。

由于PyTorch、MindSpore下蒸馏实现存在差异，对DistillDualModels实例的使用也存在如下区别。
- PyTorch下，DistillDualModels实例前向传播后返回三个数据，分别为soft label计算得到的loss、student模型的原始输出、teacher模型的原始输出。若需要获取hard label的loss，需用户自行根据student模型的原始输出计算，并调用DistillDualModels实例的get_total_loss()方法，获取soft label和hard label的综合loss。
- MindSpore下会自动计算所有loss，无需手动计算hard label。

### 函数原型
```python
get_distill_model(teacher, student, config)
```

### 参数说明
| 参数名| 输入/返回值 | 含义 | 使用限制 |
| ------ | ------ | ------ | ------ |
| teacher | 输入 | 教师模型。| 必选。<br>数据类型：MindSpore模型或PyTorch模型。 |
| student | 输入 | 学生模型。|  必选。<br>数据类型：MindSpore模型或PyTorch模型。 |
| config | 输入 | 蒸馏的配置。|  必选。<br>数据类型：KnowledgeDistillConfig对象。 |

### 调用示例
```python
from msmodelslim.common.knowledge_distill.knowledge_distill import KnowledgeDistillConfig, get_distill_model
#定义配置
distill_config = KnowledgeDistillConfig()
distill_config.set_hard_label(0.5, 0) \
  .add_inter_soft_label({
    't_module': 'uniter.encoder.encoder.blocks.11.output',
    's_module': 'uniter.encoder.encoder.blocks.5.output',
    't_output_idx': 0,
    's_output_idx': 0,
    "loss_func": [{"func_name": "KDCrossEntropy",
             "func_weight": 1}],
    'shape': [2048]
  }) 
#传入参数，返回蒸馏模型
distill_model = get_distill_model(teacher_model, student_model, distill_config)
```