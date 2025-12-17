## get_quant_params

### 功能说明 
用于获取Conv2dQuantizer, LinearQuantizer的输入和权重的量化参数，该量化参数用于后续推理。包括量化输入尺度 (input_scale)、量化输入偏移量(input_offset)、量化权重尺度(weight_scale)、量化权重偏移量(weight_offset) 以及量化权重 (quant_weight)。最后，将这些参数以字典的形式返回。

### 函数原型
```python
get_quant_params()
```

### 参数说明
| 参数名| 输入/返回值 | 含义 | 使用限制 |
| ------ | ------ | ------ | ------ |
| input_scale | 返回值 | 输入量化尺度。| 数据类型：dict。 |
| input_offset | 返回值 | 输入量化偏移量。| 数据类型：dict。 |
| weight_scale | 返回值 | 量化权重尺度。| 数据类型：dict。 |
| weight_offset | 返回值 | 量化权重偏移量。| 数据类型：dict。 |
| quant_weight | 返回值 | 量化权重。| 数据类型：dict。 |


### 调用示例
```python
from msmodelslim.pytorch.quant.ptq_tools import QuantConfig, Calibrator
disable_names = []
input_shape = [1, 3, 224, 224]
quant_config = QuantConfig(disable_names=disable_names, amp_num=0, input_shape=input_shape)
calibrator = Calibrator(model, quant_config)
calibrator.run()
input_scale, input_offset,weight_scale, weight_offset, quant_weight = calibrator.get_quant_params()
```