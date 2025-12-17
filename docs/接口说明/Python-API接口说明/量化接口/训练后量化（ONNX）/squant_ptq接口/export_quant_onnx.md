## export_quant_onnx

### 功能说明
量化参数配置类，通过calibrator类封装量化算法来保存量化后的onnx模型。

### 函数原型
```python
export_quant_onnx(save_path, fuse_add=True, use_external=False)
```

### 参数说明
| 参数名| 输入/返回值 | 含义 | 使用限制 |
| ------ | ------ | ------ | ------ |
| save_path | 输入 | 量化模型的存放路径。| 必选。<br>数据类型：String。|
| fuse_add | 输入 | 导出的量化模型是否融合量化bias。| 可选。<br>数据类型：bool。<br>默认为True。|
| use_external | 输入 | 是否需要使用额外数据存储模型。若模型保存体积过大(>2GB)，则需开启该参数使用额外数据存储模型。| 可选。<br>数据类型：bool。<br>默认为False。|


### 调用示例
```python
from msmodelslim.onnx.squant_ptq import OnnxCalibrator, QuantConfig 
quant_config = QuantConfig(disable_names=[],
                     quant_mode=0,
                     amp_num=0)
output_model_path="/home/xxx/Resnet50/resnet50_quant.onnx"    #根据实际情况配置 
calibrator = OnnxCalibrator(input_model_path, quant_config)
calibrator.run() 
calibrator.export_quant_onnx(output_model_path)
```