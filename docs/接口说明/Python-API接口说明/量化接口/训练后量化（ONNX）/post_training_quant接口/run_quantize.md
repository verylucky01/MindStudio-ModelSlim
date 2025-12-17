## run_quantize

### 功能说明
模型量化接口，对用户提供的模型根据配置的量化参数进行量化，并保存量化后模型。

### 函数原型
```python
run_quantize(input_model_path, output_model_path, quant_config)
```

### 参数说明
| 参数名| 输入/返回值 | 含义 | 使用限制 |
| ------ | ------ | ------ | ------ |
| input_model_path | 输入 | 待量化模型存放路径和文件名。| 必选。<br>数据类型：String。|
| output_model_path | 输入 | 量化后的模型的存放路径和文件名。| 必选。<br>数据类型：String。|
| quant_config | 输入 | 根据QuantConfig生成的量化配置实例。| 必选。<br>QuantConfig。|

### 调用示例
```python
from msmodelslim.onnx.post_training_quant import QuantConfig, run_quantize
def custom_read_data():
    calib_data = []
    # TODO 读取数据集，进行数据预处理，将数据存入calib_data
    return calib_data
calib_data = custom_read_data() 
quant_config = QuantConfig(calib_data=calib_data, amp_num=5)
input_model_path="/home/xxx/Resnet50/resnet50_pytorch.onnx"   #根据实际路径配置
output_model_path="/home/xxx/Resnet50/resnet50_quant.onnx"    #根据实际情况配置
run_quantize(input_model_path,output_model_path,quant_config)
```