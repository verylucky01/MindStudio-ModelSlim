## export_quant_onnx

### 功能说明
量化参数配置类，通过calibrator类封装量化算法来保存量化后的onnx模型。

### 函数原型
```python
export_quant_onnx(model_arch, save_path, input_names=None, fuse_add=True, save_fp=False)
```

### 参数说明
| 参数名| 输入/返回值 | 含义 | 使用限制 |
| ------ | ------ | ------ | ------ |
|model_arch|输入|模型结构名称。|必选。<br>数据类型：String。 |
|save_path|输入|量化模型的存放路径。|必选。<br>数据类型：String。 |
|input_names|输入|onnx的输入名称，有N个输入就要写N个名称。|可选。<br>数据类型：list[str]。<br>默认情况下，onnx名称默认为input.1、input.2……到input.N的顺序，数字命名根据模型输入的个数进行递增。|
|fuse_add|输入|导出的量化模型是否融合量化bias。|可选。<br>数据类型：bool。<br>默认为True。 |
|save_fp|输入|是否保留量化前onnx模型。|可选。<br>数据类型：bool。<br>默认为False。 |

### 调用示例
```python
from msmodelslim.pytorch.quant.ptq_tools import QuantConfig, Calibrator
disable_names = []
input_shape = [1, 3, 224, 224]
quant_config = QuantConfig(disable_names=disable_names, amp_num=0, input_shape=input_shape)
calibrator = Calibrator(model, quant_config)
calibrator.run()
calibrator.export_quant_onnx("model", "./output", ["input.1"])
```