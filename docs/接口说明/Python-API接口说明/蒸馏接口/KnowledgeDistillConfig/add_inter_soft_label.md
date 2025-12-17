## add_inter_soft_label

### 功能说明
KnowledgeDistillConfig类方法，配置蒸馏的soft label，即student模型和teacher模型的soft label的映射关系，非必须调用的方法。

### 函数原型
```python
add_inter_soft_label(config)
```

### 参数说明
参数名：config；数据类型：dict；包含的配置项如下所示：
|配置项 | 含义 | 使用限制 |
| --- | --- | --- |
| t_module | 用于配置teacher模型中计算soft label的layer名称。 | 必选。<br>数据类型：string。 | 
| s_module | 用于配置student模型中计算soft label的layer名称。 |必选。<br>数据类型：string。| 
| t_output_idx | 用于配置t_module输出的index。<br>若t_module存在多个输出，需要使用该参数指定用于计算loss的输出。若只有一个输出，使用0即可。 |必选。<br>数据类型：int。| 
| s_output_idx | 用于配置s_module输出的index。<br>若s_module存在多个输出，需要使用该参数指定用于计算loss的输出。若只有一个输出，使用0即可。 |必选。<br>数据类型：int。| 
| loss_func | 用于指定t_module 与s_module 的loss function，每一个loss function作为一个字典存入该list中，字典内部包含如下字段：<br>(1)func_name:loss function的名称。MindSpore和PyTorch模型配置为KDCrossEntropy。自定义：使用add_custom_loss_func方法新增loss function。<br>(2)func_weight：loss的权重。<br>(3)temperature：蒸馏的温度。<br>(4)func_param：部分loss function的参数。 |必选。<br>数据类型：list。字典内参数：<br>(1)func_name必选，数据类型：string。<br>(2)func_weight必选，数据类型：int。<br>(3)temperature可选，数据类型：float，默认值为1。<br>(4)func_param可选，数据类型：list，默认值为[]。| 
| shape | t_module[t_output_idx]的shape。<br>仅MindSpore模型需要配置。|可选。<br>数据类型：list。<br>需注意t_module[t_output_idx]与s_module[s_output_idx]的shape保持一致。| 

### 调用示例
```python
from msmodelslim.common.knowledge_distill.knowledge_distill import KnowledgeDistillConfig
distill_config = KnowledgeDistillConfig()
distill_config.set_hard_label(0.5, 0) \
  .add_inter_soft_label({
                    "t_module": "uniter.encoder.encoder.blocks.11.output",
                    "s_module": "uniter.encoder.encoder.blocks.5.output",
                    "t_output_idx": 0,
                    "s_output_idx": 0,
                    "loss_func": [{"func_name": "KDCrossEntropy",
                                   "func_weight": 1,
                                   "temperature": 1,},
                    ],
                    "shape": [2048]
  })
```