## create_quant_config

### 功能说明 
训练后量化接口，根据图的结构找到所有可量化的层，自动生成量化配置文件，并将可量化层的量化配置信息写入配置文件。

### 函数原型
```python
create_quant_config(config_file, model)
```

### 参数说明
| 参数名| 输入/返回值 | 含义 | 使用限制 |
| ------ | ------ | ------ | ------ |
| config_file | 输入 | 待生成的量化配置文件存放路径及名称。必选。| 数据类型：string。配置文件必须为.json后缀的文件。如果存放路径下已经存在该文件，则调用该接口时会覆盖已有文件。 |
| model | 输入 | 待量化的模型实例。必选。| 数据类型：MindSpore模型。 |


### 调用示例
```python
from msmodelslim.mindspore.quant.ptq_quant.create_config import create_quant_config
model = SampleModel()
config_file = "./test_config_file.json"
create_quant_config(config_file, model)
```