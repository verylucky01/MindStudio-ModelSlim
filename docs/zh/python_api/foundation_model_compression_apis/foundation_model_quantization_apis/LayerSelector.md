## LayerSelector

### 功能说明
构建用于层选择的类，通过分析模型各层的量化难度，帮助用户选择需要跳过量化的层。

### 函数原型
```python
LayerSelector(model, layer_names=None, range_method="quantile")
```

### 参数说明
| 参数名| 输入/返回值 | 含义 | 使用限制 |
| ------ | ------ | ------ | ------ |
| model | 输入 | 需要进行层选择分析的模型。| 必选。<br>数据类型：torch.nn.Module |
| layer_names | 输入 | 需要分析的层名称列表。| 可选。<br>数据类型：list。<br>默认值为None，表示分析所有线性层和卷积层。 |
| range_method | 输入 | 用于计算量化难度的方法。| 可选。<br>数据类型：str。<br>默认值为"quantile"。<br>可选值："quantile"或"std"。 |

### 主要接口说明

#### run
```python
run(calib_data)
```
| 参数名| 输入/返回值 | 含义 | 使用限制 |
| ------ | ------ | ------ | ------ |
| calib_data | 输入 | 用于分析层量化难度的校准数据。| 必选。<br>数据类型：list。<br>输入模板：\[[input1],[input2],[input3]]。 |

#### select_layers_by_threshold
```python
select_layers_by_threshold(threshold)
```
| 参数名| 输入/返回值 | 含义 | 使用限制 |
| ------ | ------ | ------ | ------ |
| threshold | 输入 | 量化难度阈值。| 必选。<br>数据类型：float。<br>取值范围：>0。 |
| 返回值 | 返回 | 量化难度超过阈值的层名称列表。| 数据类型：list。 |

#### select_layers_by_disable_level
```python
select_layers_by_disable_level(disable_level)
```
| 参数名| 输入/返回值 | 含义 | 使用限制 |
| ------ | ------ | ------ | ------ |
| disable_level | 输入 | 需要跳过量化的层级数。| 必选。<br>数据类型：int。<br>取值范围：≥0。 |
| 返回值 | 返回 | 选定层级中的层名称列表。| 数据类型：list。 |

### 调用示例
根据实际需求，使用LayerSelector分析模型各层的量化难度，并选择需要跳过量化的层。
```python
from msmodelslim.pytorch.llm_ptq.llm_ptq_tools.layer_select import LayerSelector

# 初始化LayerSelector
layer_selector = LayerSelector(model, range_method="quantile")

# 运行分析
layer_selector.run(calib_data)

# 通过阈值选择层
disable_names = layer_selector.select_layers_by_threshold(threshold=1.0)

# 或通过层级选择层
disable_names = layer_selector.select_layers_by_disable_level(disable_level=10)

# 将选择的层用于量化配置
quant_config = QuantConfig(disable_names=disable_names)
calibrator = Calibrator(model, quant_config, calib_data=calib_data)
calibrator.run() 
```