## run()

### 功能说明
运行量化算法，初始化OnnxCalibrator后通过run()函数来执行量化。

### 函数原型
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