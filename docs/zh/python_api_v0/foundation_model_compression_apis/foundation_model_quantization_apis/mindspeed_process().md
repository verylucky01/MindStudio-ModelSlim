## process()

### 功能说明
使用校准集，执行异常值抑制过程，修改模型中的权重，提升后续模型量化精度，无传入值。

### 函数原型
```python
process()
```

### 调用示例
根据实际需求，在AntiOutlierConfig初始化中完成所有参数的配置。
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