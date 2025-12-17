## export_param

### 功能说明 
用于将Conv2dQuantizer, LinearQuantizer的输入和权重的量化参数保存为npy文件，该量化参数用于后续推理。包括量化输入尺度 (input scale)、量化输入偏移量(input offset)、量化权重尺度(weight_scale)、量化权重偏移量(weight_offset) 以及量化权重 (quant weight)。
### 函数原型
```python
export_param(save_path)
```

### 参数说明
| 参数名| 输入/返回值 | 含义 | 使用限制 |
| ------ | ------ | ------ | ------ |
| save_path | 返回值 | 量化参数的保存路径。| 必选。数据类型：string。 |



### 调用示例
```python
from msmodelslim.pytorch.quant.ptq_tools import QuantConfig, Calibrator
disable_names = []
input_shape = [1, 3, 224, 224]
quant_config = QuantConfig(disable_names=disable_names, amp_num=0, input_shape=input_shape)
calibrator = Calibrator(model, quant_config)
calibrator.run()
calibrator.export_param("./output")
```