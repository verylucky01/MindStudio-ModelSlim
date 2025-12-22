## CalibratorAdapter

### 功能说明
针对MindSpeed-LLM模型的量化参数配置类，继承自Calibrator，对外接口与Calibrator一致，不支持cpu执行量化。

### 继承关系
CalibratorAdapter继承自Calibrator类，保持了与Calibrator相同的外部接口，包括：
- run()：执行量化流程
- save()：保存量化后的模型和权重

### 函数原型
```python
CalibratorAdapter(model, cfg: QuantConfig, calib_data=None, disable_level='L0', all_tensors=None, mix_cfg: Optional[dict] = None)
```

### 参数说明
| 参数名| 输入/返回值 | 含义 | 使用限制 |
| ------ | ------ | ------ | ------ |
| model | 输入 | 模型。| 必选。<br>数据类型：PyTorch模型。 |
| cfg | 输入 | 已配置的QuantConfig类。| 必选。<br>数据类型：QuantConfig。 |
| calib_data | 输入 | LLM大模型量化校准的数据，输入真实数据用于Label-Free量化。| 可选。<br>数据类型：object。<br>默认值为None，为Data-Free场景，Label-Free场景必须输入。<br>输入模板：\[[input1],[input2],[input3]]。|
| disable_level | 输入 | 自动回退等级，在模型精度损失大时，可以适当提升等级，但回退层数不可以大于模型总层数。| 可选。<br>数据类型：string。<br>配置示例如下：(1)'L0':默认值，不执行回退。(2)'L1'：回退1层。(3)'L2'：回退2层。(3)'L3'：回退3层。(4)'L4'：回退4层。(5)'L5'：回退5层。<br>以此类推。|
| all_tensors | 输入 | 用于逐层量化校准的{name:tensor}。| 可选。<br>数据类型：dict。<br>默认值为None，采用默认配置即可。|
| mix_cfg | 输入 | 混合量化配置，指定{**nn.Module**的通配符或层名：量化类型}。<br>关键：**通配符区分大小写。并且实现采用Python的标准库fnmatch，因此相关用法请参考fnmatch**。| 可选。<br>数据类型：dict。<br>默认值为None。<br>配置示例如下：{"\*down\*": "w8a16", "\*": "w8a8"}<br>**量化类型（字典的值）必须是内部可识别的配置，否则会报错。**<br>当下支持w8a8，w8a16，w8a8_dynamic。

### 调用示例
根据实际需求，在QuantConfig初始化中完成所有参数的配置。
```python
from msmodelslim.pytorch.mindspeed_adapter import ModelAdapter, CalibratorAdapter
model = ModelAdapter(model)
quant_config = QuantConfig(dev_type='npu', pr=0.5, mm_tensor=False)
calibrator = CalibratorAdapter(model, quant_config, calib_data=dataset_calib, disable_level='L0')
calibrator.run()  # 与Calibrator相同的用法
calibrator.save(quant_weight_save_path)  # 与Calibrator相同的用法
```
