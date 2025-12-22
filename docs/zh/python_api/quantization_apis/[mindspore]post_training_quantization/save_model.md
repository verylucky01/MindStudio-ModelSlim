## save_model

### 功能说明 
训练后量化接口，根据修改后的图结构，插入AscendQuant、AscendDequant等算子，将模型保存为量化后的离线模型文件。

### 函数原型
```python
save_model(file_name, quantized_model, *input_data, file_format='AIR')
```

### 参数说明
| 参数名| 输入/返回值 | 含义 | 使用限制 |
| ------ | ------ | ------ | ------ |
| file_name | 输入 | 模型存放路径和文件名。必选。| 数据类型：string。 |
| quantized_model | 输入 | 通过quantize_model接口生成的量化后的模型。必选。| 数据类型：MindSpore模型。 |
| input_data | 输入 | 用户网络输入数据。必选。| 数据类型：MindSpore的Tensor。需要与MindSpore模型的输入保持一致。 |
| file_format | 输入 | 离线模型的格式。必选。| 可选值：AIR和MINDIR。默认值为AIR。 |


### 调用示例
```python
from msmodelslim.mindspore.quant.ptq_quant.save_model import save_model
file_name = "./save_model"
input_data = ms.Tensor(np.random.uniform(size=[1, 3, 224, 224]), dtype=mstype.float32)
calibrate_model = quantize_model(config_file, model, input_data)
save_model(file_name, calibrate_model, input_data, file_format="MINDIR")
```