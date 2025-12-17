## AntiOutlierConfig

### 功能说明
构建用于异常值抑制的config。

### 函数原型
```python
AntiOutlierConfig(w_bit=8, a_bit=8, anti_method="m2", dev_type="cpu", dev_id=None, w_sym=True)
```

### 参数说明
| 参数名| 输入/返回值 | 含义 | 使用限制 |
| ------ | ------ | ------ | ------ |
| w_bit | 输入 | 权重量化bit。| 可选。<br>数据类型：int。<br>默认为8，不支持修改。 |
| a_bit | 输入 | 激活层量化bit。| 可选。<br>数据类型：int。<br>默认为8。<br>anti_method为m3时，支持修改为16 |
| anti_method | 输入 | 异常值抑制anti_outlier使用方法。| 可选。<br>数据类型：string。<br>默认为m2，可选值'm1'或'm2'或'm3'或'm4'或'm5'或'm6'。<br>（1）'m1':异常值抑制方式1。<br>（2）'m2':异常值抑制方式2，推荐使用。<br>（3）'m3':AWQ算法。<br>（4）'m4':smooth优化算法 。<br>（5）'m5':CBQ量化算法。<br>（6）'m6':Flex smooth量化算法。<br>说明：m4方式不支持telechat模型进行量化的场景。m3方式处理MOE模型时，不对专家结构做任何处理。m2方式不支持MOE模型w8a8-pertoken场景，当前已适配qwen-vl和llava-v1.5-7b多模态模型。 |
| disable_anti_names | 输入 | 指定某些层不进行异常值抑制。| 可选。<br>数据类型：list。<br>默认为[]。<br>仅在anti_method设置为m6时生效，仅支持传入o层。 |
| flex_config | 输入 | m6方式下的配置文件。| 可选。<br>数据类型：dict。<br>仅在 anti_method 设置为 m6 时生效。包含两个配置项：alpha 和 beta，均为 float 类型，取值范围为 [0, 1]，默认值为 None。用于控制算法的平滑程度。如果 alpha 和 beta 均指定为具体数值，则直接使用这些值；如果任一值未指定（为 None），算法将自动进行寻优以计算最优的 alpha 和 beta 值，用于异常值抑制。|
| dev_type | 输入 | device类型。| 可选。<br>数据类型：string。<br>可选值：['cpu', 'npu']，默认为'cpu'。 |
| dev_id | 输入 | Device ID。| 可选。<br>数据类型：int。<br>默认值为None。<br>仅在“dev_type”配置为“npu”时生效。“dev_id”指定的Device ID优先级高于环境变量配置的Device ID。 |
| w_sym | 输入 | 权重是否对称量化。| 可选。<br>数据类型：bool。<br>默认为True。<br>anti_method设置为m3时，可以选择为False，需与QuantConfig中的w_sym参数设置一致。 |

### 调用示例
根据实际需求，在QuantConfig初始化中完成所有参数的配置。
```python
from msmodelslim.pytorch.llm_ptq.llm_ptq_tools import Calibrator
from msmodelslim.pytorch.llm_ptq.anti_outlier import AntiOutlier, AntiOutlierConfig
anti_config = AntiOutlierConfig(anti_method="m2")
anti_outlier = AntiOutlier(model, calib_data=dataset_calib, cfg=anti_config)
anti_outlier.process() 
calibrator = Calibrator(model, quant_config, calib_data=dataset_calib, disable_level='L0') 
calibrator.run(int_infer=False) 
calibrator.save(quant_weight_save_path)
```

anti_method=m6时调用示例：
```python
from msmodelslim.pytorch.llm_ptq.llm_ptq_tools import Calibrator
from msmodelslim.pytorch.llm_ptq.anti_outlier import AntiOutlier, AntiOutlierConfig
# 指定o层不进行异常值抑制
keys = ['.o_proj']
disable_names = []
for name, mod in model.named_modules():
    if isinstance(mod, torch.nn.Linear):
        for key in keys:
            if key in name:
                disable_names.append(name)
# 若alpha和beta均指定为具体数值，则直接使用指定值进行配置
anti_config = AntiOutlierConfig(anti_method='m6',
                                disable_anti_names=disable_names,
                                flex_config={'alpha': 0.75,
                                             'beta': 0.1})
anti_outlier = AntiOutlier(model, calib_data=dataset_calib, cfg=anti_config)
anti_outlier.process() 
calibrator = Calibrator(model, quant_config, calib_data=dataset_calib, disable_level='L0') 
calibrator.run(int_infer=False) 
calibrator.save(quant_weight_save_path)
```