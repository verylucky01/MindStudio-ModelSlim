## OnnxCalibrator

### 功能说明
量化参数配置类，通过Calibrator类封装量化算法。

### 函数原型
```python
OnnxCalibrator(input_model, cfg: QuantConfig, calib_data=None)
```

### 参数说明
| 参数名| 输入/返回值 | 含义 | 使用限制 |
| ------ | ------ | ------ | ------ |
| input_model | 输入 | 待量化模型存放路径和文件名。| 必选。<br>数据类型：String。|
| cfg | 输入 | 已配置的QuantConfig类。| 必选。<br>数据类型：QuantConfig。|
| calib_data | 输入 | 模型训练数据，可输入真实数据用于Label-Free量化，也可输入虚拟数据来实现Label-Free量化。| 可选。<br>数据类型：list，默认值为[]。<br>对于单输入模型，配置\[[input1]]，多输入模型，配置\[[input1,input2,input3]]。<br>(1)如果是单输入场景，可以不输入数据，在模型支持单个float格式输入且指定了input_shape时，会自动调用Label-Free量化流程。<br>(2)针对多个输入或者需要自定义输入格式的模型，用户必须手动输入数据来实现Label-Free量化。模板示例：calib_data = \[[np.random.random(size=(1, 3, 127, 127)).astype(np.float32), np.random.random(size=(1, 3, 255, 255)).astype(np.float32)]]。|

### 调用示例
```python
from msmodelslim.onnx.squant_ptq import OnnxCalibrator, QuantConfig 
quant_config = QuantConfig(disable_names=[],
                     quant_mode=0,
                     amp_num=0)
input_model_path="/home/xxx/Resnet50/resnet50_pytorch.onnx"   #根据实际路径配置
calib = OnnxCalibrator(input_model_path, quant_config)
```