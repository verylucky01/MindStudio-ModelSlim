## AntiOutlierAdapter

### 功能说明
构建用于异常值抑制的类，并将模型，异常值抑制配置，校准数据等传入，输入接口与llm_ptq的AntiOutlier一致，不支持cpu执行。

### 函数原型
```python
AntiOutlierAdapter(model, calib_data=None, cfg=None, norm_class_name = None)
```

### 参数说明
| 参数名| 输入/返回值 | 含义 | 使用限制 |
| ------ | ------ | ------ | ------ |
| model | 输入 | 用于大模型离群值抑制的模型。| 必选。<br>数据类型：PyTorch模型。 |
| calib_data | 输入 | 用于离群值抑制的校准数据。| 可选。<br>数据类型：object。<br>默认值为None。<br>输入模板：\[[input1],[input2],[input3]]。 |
| cfg | 输入 | 已配置的AntiOutlierConfig类。| 可选。<br>数据类型：Config。 |
| norm_class_name | 输入 | 用户自定义的norm类名。| 可选。<br>数据类型：str。<br>默认为None，若系统自动识别norm失败，则需要用户手动输入自定义的norm类名，例如norm_class_name = 'LlamaRMSNorm'。 |

### 调用示例
根据实际需求，在QuantConfig初始化中完成所有参数的配置。
```python
from msmodelslim.pytorch.llm_ptq.anti_outlier import AntiOutlierConfig
from msmodelslim.pytorch.mindspeed_adapter import ModelAdapter, AntiOutlierAdapter, CalibratorAdapter
model = ModelAdapter(model)
anti_config = AntiOutlierConfig(anti_method="m5", dev_type='npu')
anti_outlier = AntiOutlierAdapter(model, calib_data=dataset_calib, cfg=anti_config)
anti_outlier.process() 
calibrator = CalibratorAdapter(model, quant_config, calib_data=dataset_calib, disable_level='L0') 
calibrator.run(int_infer=False) 
calibrator.save(quant_weight_save_path)
```
