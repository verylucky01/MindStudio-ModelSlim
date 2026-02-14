## ModelAdapter

### 功能说明
MindSpeed的模型适配器，将MindSpeed-LLM模型转化为msModelSlim可以量化的LLM模型。

### 函数原型
```python
ModelAdapter(model:nn.Module, dev_type='npu', forward_step=None, prefix='model.')
```

### 参数说明
| 参数名| 输入/返回值 | 含义 | 使用限制 |
| ------ | ------ | ------ | ------ |
| model | 输入 | 模型。| 必选。<br>数据类型：MindSpeed-LLM模型。 |
| dev_type | 输入 | 设备的device类型。| 可选。<br>数据类型：string。<br>可选值：['npu', 'gpu'], 默认值为'npu' |
| forward_step | 输入 | 启动MindSpeed-LLM模型的函数对象，当无法正常拉起模型时可以自定义一个启动器。| 可选。<br>数据类型：函数对象。<br>默认值为None，会使用自带的模型拉起方式|
| prefix | 输入 | 模型state_dict的前缀名。| 可选。<br>数据类型：string。<br>可选值：["model.", "model.module."]<br>默认值为"model."，当模型的state_dict的keys和模型本身结构不一致时，需要通过调整prefix来保证一致性，否则会影响模型保存的正确执行。例如，model.state_dict()获取到的权重为"language_model.output_layer.weight"，而该权重实际模型路径为"model.language_model.output_layer.weight"，此时应将该参数设置为'model.'，以修正前缀名称。|

### 调用示例
```python
from msmodelslim.pytorch.mindspeed_adapter import ModelAdapter
from megatron.inference.text_generation import generate
class GenerateForward:
    def __call__(self, model, x):
        return generate(model, x, tokens_to_generate=1)
model = ModelAdapter(model, forward_step=GenerateForward(), prefix='model.')
```
