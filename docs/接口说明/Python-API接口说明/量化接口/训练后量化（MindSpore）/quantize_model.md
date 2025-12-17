## quantize_model

### 功能说明 
训练后量化接口，根据用户设置的量化配置文件对图结构进行量化处理，该函数在config_file指定的层插入权重量化层，完成权重量化，并插入数据量化层，将修改后的网络存为新的模型文件。

### 函数原型
```python
quantize_model(config_file, model, *input_data)
```

### 参数说明
| 参数名| 输入/返回值 | 含义 | 使用限制 |
| ------ | ------ | ------ | ------ |
| config_file | 输入 | 通过create_quant_config接口生成的量化配置文件，用于指定模型中量化层的配置情况。必选。| 数据类型：string。 |
| model | 输入 | 已加载过训练参数MindSpore网络模型。必选。| 数据类型：MindSpore模型。 |
| input_data | 输入 | 用户网络输入数据。必选。| 数据类型：MindSpore的Tensor。需要与MindSpore模型的input保持一致的shape。 |


### 调用示例
```python
from msmodelslim.mindspore.quant.ptq_quant.quantize_model import quantize_model
config_file = "./test_config_file.json"
model = SampleModel()
input_data = ms.Tensor(np.random.uniform(size=[1, 3, 224, 224]), dtype=mstype.float32)
calibrate_model = quantize_model(config_file, model, input_data)
```