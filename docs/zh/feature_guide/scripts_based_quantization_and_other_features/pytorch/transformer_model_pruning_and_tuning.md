# Transformer类模型权重剪枝调优

## 简介
msModelSlim工具提供了API方式的Transformer类模型权重剪枝调优，可将模型权重进行裁剪，并加载到同一模型结构下的小模型中。用户只需提供同一模型结构下小模型(通过配置较小初始化参数得到的模型实例，例如Bert模型中缩小intermediate_size和num_hidden_layers参数)和原始模型权重文件，即可调用剪枝API完成模型权重的剪枝。

## 使用前准备
目前支持MindSpore和PyTorch框架下Transformer类模型的剪枝调优，执行剪枝调优前需参考[安装指南](../../../install_guide.md)完成开发环境配置。

- 注意：该功能仅支持 PyTorch 2.0.0 以上版本。

模型剪枝期间，用户可手动配置参数对预训练模型的权重进行裁剪，并将裁剪后的权重加载至小模型中，获取一个权重加载完毕的Transformer模型。剪枝后模型不保障精度，需要进行一定的训练来提升精度，例如通过模型蒸馏进行训练。

## 功能介绍

### 操作步骤 

以下步骤以PyTorch框架的Transformer类模型为例，MindSpore框架的模型仅在调用部分接口时，入参配置有所差异，使用时请参照具体的API接口说明。

1. 用户自行准备同一种模型结构下的原始模型实例（待剪枝模型）和原始模型权重文件。本样例以Bert为例，在ModelZoo搜索下载Bert代码和原始模型权重文件。

2. 新建待剪枝模型的Python脚本，例如test_prune_model.py。编辑test_prune_model.py文件，导入如下接口。剪枝API接口说明请参考剪枝接口。
```python
from msmodelslim.common.prune.transformer_prune.prune_model import PruneConfig
from msmodelslim.common.prune.transformer_prune.prune_model import prune_model_weight
```

3. （可选）调整日志输出等级，启动调优任务后，将打屏显示设置级别的日志信息。[日志级别说明](../../../python_api/common_apis.md#参数说明)
```python
from msmodelslim import set_logger_level
set_logger_level("info")        #根据实际情况配置
```

4. 使用PruneConfig接口自定义配置剪枝的步骤和block，请参考PruneConfig进行配置。
```python
prune_config = PruneConfig()
prune_config.set_steps(['prune_blocks', 'prune_bert_intra_block']). \
    add_blocks_params(pattern="bert.encoder.layer.(\d+).",layer_id_map={0: 0, 1: 2, 2: 4, 3: 6, 4: 8, 5: 10, 6: 11})
```
- 说明：若set_steps方法中配置的剪枝步骤包含“prune_blocks”，则必须调用“add_blocks_params”方法进行配置。

5. 使用prune_model_weight接口调用剪枝配置项修改预训练的模型权重，并将剪枝后的权重载入小模型中，小模型通过较小的初始化参数生成。
以Bert为例，初始化较小模型时，需提前修改bert_config下的json配置，例如intermediate_size参数改小为1536，num_hidden_layers 参数改小为7。修改后在Python脚本中导入如下内容进行配置。
```python
import modeling # 导入bert模型
bert_config = modeling.BertConfig.from_json_file(bert_config_file) # 载入bert配置，初始化较小的模型。
bert_model = modeling.BertForQuestionAnswering(bert_config) # 实例化bert模型
prune_model_weight(bert_model, prune_config, weight_file_path = "/home/xxx/xxx.pt")   #model根据实际情况配置待剪枝模型实例，weight_file_path根据实际情况配置原模型的权重文件
```
MindSpore模型的权重文件需为ckpt格式，PyTorch框架的权重文件需为pt/pth/pkl/bin格式，具体请参考prune_model_weight进行配置。

6. 启动模型剪枝调优任务，将原始权重进行裁剪并载入小模型中。
```python
python3 test_prune_model.py
```
