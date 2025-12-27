# 模型蒸馏

## 简介

msModelSlim工具支持API方式的蒸馏调优。蒸馏调优时，用户只需要提供teacher模型、student模型和数据集，调用API接口完成模型的蒸馏调优过程。

模型蒸馏期间，用户可将原始Transformer模型、配置较小参数的Transformer模型分别作为teacher和student进行知识蒸馏。通过手动配置参数，返回一个待蒸馏的DistillDualModels模型实例，用户对其进行训练。训练完毕后，从DistillDualModels模型实例获取训练后的student模型，即通过蒸馏训练后的模型。

## 使用前准备
目前支持MindSpore和PyTorch框架下Transformer类模型的蒸馏调优，执行前需参考[安装指南](../../../install_guide.md)完成开发环境配置。

## 功能介绍

以下步骤以PyTorch框架的模型为例，MindSpore框架的模型仅在调用部分接口时，入参配置有所差异，使用时请参照具体的API接口说明。

1. 用户自行准备原始Transformer模型、配置较小参数的Transformer模型，分别作为模型蒸馏调优的teacher模型和student模型。本样例以Bert为例，在ModelZoo搜索下载Bert代码和原模型权重文件。

2. 新建待蒸馏模型的Python脚本，例如distill_model.py。编辑distill_model.py文件，导入如下接口。蒸馏API接口说明请参考蒸馏接口。
```
from msmodelslim.common.knowledge_distill.knowledge_distill import KnowledgeDistillConfig, get_distill_model
```

3. （可选）调整日志输出等级，启动调优任务后，将打屏显示蒸馏调优的日志信息。

```
from msmodelslim import set_logger_level
set_logger_level("info")        #根据实际情况配置
```

4. 使用KnowledgeDistillConfig接口自定义配置模型蒸馏的参数，请参考KnowledgeDistillConfig进行配置。

```
distill_config = KnowledgeDistillConfig()
distill_config.add_output_soft_label({
                "t_output_idx": 1,
                "s_output_idx": 1,
                "loss_func": [{"func_name": "KDCrossEntropy",
                               "func_weight": 1,
                               "temperature": 1}]})
```

5. 使用get_distill_model接口调用蒸馏配置项并返回一个待蒸馏的DistillDualModels模型实例，请参考get_distill_model进行配置。teacher_model、student_model为Bert的实例，通过修改bert_configs下的json配置，初始化不同大小的Bert模型。
```
distill_model = get_distill_model(teacher_model, student_model, distill_config)   #请传入teacher模型、student模型的实例
```

6. 用户自行对待蒸馏的DistillDualModels模型实例进行训练，可参考teacher、student模型的训练脚本、MindSpore官网或PyTorch官网进行训练。以Bert为例，参考原始训练代码run_squad.py进行如下重点信息修改，并执行命令进行训练。

- 将原始代码中model = modeling.BertForQuestionAnswering(config)改为model = distill_model.student_model，从而为student模型设置optimizer。
- 将原始代码中start_logits, end_logits = model(input_ids, segment_ids, input_mask)改为loss, student_outputs, teacher_outputs = distill_model (input_ids, segment_ids, input_mask)，并注释原始的loss计算部分，从而对 DistillDualModels模型实例进行训练。
训练完成后，可以使用get_student_model方法，获取训练后的student模型（MindSpore框架的模型使用get_student_model方法后，无法再次对DistillDualModels模型实例进行训练）。

7. 训练完成后，可以使用get_student_model方法，获取训练后的student模型（MindSpore框架的模型使用get_student_model方法后，无法再次对DistillDualModels模型实例进行训练）。
```
student_model = distill_model.get_student_model()
```

8. 启动模型蒸馏调优任务，将获取一个训练后的student模型。
```
python3 distill_model.py
```