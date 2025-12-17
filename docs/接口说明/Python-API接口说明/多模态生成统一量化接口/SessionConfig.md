# 量化会话配置
## SessionConfig

### 功能说明
量化会话配置类，用于配置量化相关的参数、校准数据以及运行设备。

### 类原型
```python
class SessionConfig(BaseModel):
    processor_cfg_map: Dict[str, BaseModel] = {}
    calib_data: Optional[List[Any]] = None
    device: str = 'cpu'
```

### 参数说明
| 参数名| 输入/返回值 | 含义 | 使用限制 |
| ------ | ------ | ------ | ------ |
| processor_cfg_map | 输入 | 量化处理器配置映射图 | 必选。<br>数据类型：字典。默认为{}，使用量化功能时，至少应配置一个量化处理器（如W8A8ProcessorConfig），不可单独配置保存处理器（SaveProcessorConfig）。<br>每个键、值对应一个量化处理器名称和量化处理器配置类，当前可选量化处理器名称: ['m3', 'm4', 'm6', 'w8a8', 'w8a8_dynamic', 'w8a8_timestep', 'fa3', 'save']，与可选量化处理器配置[M3ProcessorConfig, M4ProcessorConfig, M6ProcessorConfig, W8A8ProcessorConfig, W8A8DynamicProcessorConfig, W8A8TimeStepProcessorConfig, FA3ProcessorConfig, SaveProcessorConfig]一一对应，其中'fa3'需要搭配'w8a8_dynamic'一起使用。|
| calib_data | 输入 | 异常值抑制和量化校准数据 | 可选。<br>数据类型：列表。默认值为None，为Data-Free场景，Label-Free场景必须输入。多模态生成模型量化场景中，需要提前dump校准数据并加载后传入作为calib_data。|
| device | 输入 | 量化过程运行设备 | 可选。<br>数据类型：字符串。默认值为'cpu'，可选值：['cpu', 'npu']。|

### 调用示例
```python
import torch
from ascend_utils.common.security.pytorch import safe_torch_load
from msmodelslim.quant.session.session import W8A8ProcessorConfig, W8A8QuantConfig, SaveProcessorConfig
from msmodelslim.quant.session.session import SessionConfig

session_config = SessionConfig(
    processor_cfg_map={
        "w8a8": W8A8ProcessorConfig(
            cfg=W8A8QuantConfig(
                act_method='minmax'
            ),
            disable_names=[]
        ),
        "save": SaveProcessorConfig(
            output_path="./"
            safetensors_name=None,
            json_name=None,
            save_type=['safe_tensor'],
            part_file_size=None
        )
    },
    calib_data=safe_torch_load("calib_data.pth"),
    device="npu"
)
```

## W8A8ProcessorConfig

### 功能说明
W8A8量化处理器配置类，用于配置W8A8量化处理器相关的参数。

### 类原型
```python
class W8A8ProcessorConfig(BaseModel):
    cfg: W8A8QuantConfig
    disable_names: list
```

### 参数说明
| 参数名| 输入/返回值 | 含义 | 使用限制 |
| ------ | ------ | ------ | ------ |
| cfg | 输入 | W8A8量化配置 | 必选。<br>数据类型：W8A8QuantConfig，W8A8量化配置类。|
| disable_names | 输入 | 回退层 | 必选。<br>数据类型：列表。列表中每个元素为回退层名称。|

### 调用示例
```python
from msmodelslim.quant.session.session import W8A8ProcessorConfig, W8A8QuantConfig

w8a8_processor_cfg = W8A8ProcessorConfig(
    cfg=W8A8QuantConfig(
        act_method='minmax'
    ),
    disable_names=[]
)
```

## W8A8QuantConfig

### 功能说明
W8A8量化配置类，用于配置W8A8量化相关的参数。

### 类原型
```python
class W8A8QuantConfig(BaseModel):
    act_method: str = 'minmax'
```

### 参数说明
| 参数名| 输入/返回值 | 含义 | 使用限制 |
| ------ | ------ | ------ | ------ |
| act_method | 输入 | 激活值量化方法 | 可选。<br>数据类型：字符串。可选值: ['minmax', 'histogram', 'mixed']，分别对应MinMax激活量化、Histogram直方图激活量化、MinMax与Histogram混合的激活量化。|

### 调用示例
```python
from msmodelslim.quant.session.session import W8A8QuantConfig

w8a8_quant_cfg = W8A8QuantConfig(
    act_method='minmax'
)
```

## W8A8DynamicProcessorConfig

### 功能说明
W8A8动态量化处理器配置类，用于配置W8A8动态量化处理器相关的参数。

### 类原型
```python
class W8A8DynamicProcessorConfig(BaseModel):
    cfg: W8A8DynamicQuantConfig
    disable_names: list
```

### 参数说明
| 参数名| 输入/返回值 | 含义 | 使用限制 |
| ------ | ------ | ------ | ------ |
| cfg | 输入 | W8A8动态量化配置 | 必选。<br>数据类型：W8A8DynamicQuantConfig，W8A8动态量化配置类。|
| disable_names | 输入 | 回退层 | 必选。<br>数据类型：列表。列表中每个元素为回退层名称。|

### 调用示例
```python
from msmodelslim.quant.session.session import W8A8DynamicProcessorConfig, W8A8DynamicQuantConfig

w8a8dynamic_processor_cfg = W8A8DynamicProcessorConfig(
    cfg=W8A8DynamicQuantConfig(
        act_method='minmax'
    ),
    disable_names=[]
)
```

## W8A8DynamicQuantConfig

### 功能说明
W8A8动态量化配置类，用于配置W8A8动态量化相关的参数。

### 类原型
```python
class W8A8DynamicQuantConfig(BaseModel):
    act_method: str = 'minmax'
```

### 参数说明
| 参数名| 输入/返回值 | 含义 | 使用限制 |
| ------ | ------ | ------ | ------ |
| act_method | 输入 | 激活值量化方法 | 可选。<br>数据类型：字符串。可选值: ['minmax', 'histogram', 'mix']，分别对应MinMax激活量化、Histogram直方图激活量化、MinMax与Histogram混合的激活量化。|

### 调用示例
```python
from msmodelslim.quant.session.session import W8A8DynamicQuantConfig

w8a8dynamic_quant_cfg = W8A8DynamicQuantConfig(
    act_method='minmax'
)
```

## W8A8TimeStepProcessorConfig

### 功能说明
W8A8时间步量化处理器配置类，用于配置W8A8时间步量化处理器相关的参数。

### 类原型
```python
class W8A8TimeStepProcessorConfig(BaseModel):
    cfg: W8A8TimeStepQuantConfig
    disable_names: list
    timestep_sep: int
```

### 参数说明
| 参数名| 输入/返回值 | 含义 | 使用限制 |
| ------ | ------ | ------ | ------ |
| cfg | 输入 | W8A8时间步量化配置 | 必选。<br>数据类型：W8A8TimeStepQuantConfig，W8A8时间步量化配置类。|
| disable_names | 输入 | 回退层 | 必选。<br>数据类型：列表。列表中每个元素为回退层名称。|
| timestep_sep | 输入 | 时间步量化的动静态量化分割阈值 | 必选。<br>数据类型：整数。通常设置为多模态生成模型视图生成中总推理时间步的一半。|

### 调用示例
```python
from msmodelslim.quant.session.session import W8A8TimeStepProcessorConfig, W8A8TimeStepQuantConfig

w8a8timestep_processor_cfg = W8A8TimeStepProcessorConfig(
    cfg=W8A8TimeStepQuantConfig(
        act_method='minmax'
    ),
    disable_names=[],
    timestep_sep=25
)
```

## W8A8TimeStepQuantConfig

### 功能说明
W8A8时间步量化配置类，用于配置W8A8时间步量化相关的参数。

### 类原型
```python
class W8A8TimeStepQuantConfig(BaseModel):
    act_method: str = 'minmax'
```

### 参数说明
| 参数名| 输入/返回值 | 含义 | 使用限制 |
| ------ | ------ | ------ | ------ |
| act_method | 输入 | 激活值量化方法 | 可选。<br>数据类型：字符串。可选值: ['minmax', 'histogram', 'mix']，分别对应MinMax激活量化、Histogram直方图激活量化、MinMax与Histogram混合的激活量化。|

### 调用示例
```python
from msmodelslim.quant.session.session import W8A8TimeStepQuantConfig

w8a8timestep_quant_cfg = W8A8TimeStepQuantConfig(
    act_method='minmax'
)
```

## FA3ProcessorConfig

### 功能说明
FA3量化处理器配置类，用于配置FA3量化处理器相关的参数。

### 类原型
```python
class FA3ProcessorConfig(BaseModel):
    pass
```

### 调用示例
```python
from msmodelslim.quant.session.session import FA3ProcessorConfig

fa3_processor_cfg = FA3ProcessorConfig()
```

## SaveProcessorConfig

### 功能说明
量化保存处理器配置类，用于配置量化保存处理器相关的参数。

### 类原型
```python
class SaveProcessorConfig(BaseModel):
    output_path: str
    safetensors_name: Optional[str] = None
    json_name: Optional[str] = None
    save_type: list = ['safe_tensor']
    part_file_size: Optional[int] = None
```

### 参数说明
| 参数名| 输入/返回值 | 含义 | 使用限制 |
| ------ | ------ | ------ | ------ |
| output_path | 输入 | 量化保存路径 | 必选。<br>数据类型：字符串。无默认值。|
| safetensors_name | 输入 | 量化权重safetensors命名 | 可选。<br>数据类型：字符串。默认值为None，根据量化类型生成，例如quant_model_weight_w8a8.safetensors。|
| json_name | 输入 | 量化权重描述文件json命名 | 可选。<br>数据类型：字符串。默认值None，根据量化类型生成，例如quant_model_description_w8a8.json。|
| save_type | 输入 | 量化权重保存格式 | 可选。<br>数据类型：列表，元素为字符串。默认值['safe_tensor']，多模态生成模型量化场景下默认采用safetensors格式。|
| part_file_size | 输入 | 保存成safetensors权重文件时，进行分片保存时，每个部分的大小，单位为GB| 可选。<br>数据类型：整型。默认值为None，不启用分片保存的功能。否则将会按照用户设置值进行分片，实际保存的权重可能略大于设置的值。|

### 调用示例
```python
from msmodelslim.quant.session.session import SaveProcessorConfig

save_processor_cfg = SaveProcessorConfig(
    output_path="./",
    safetensors_name=None,
    json_name=None,
    save_type=['safe_tensor'],
    part_file_size=None
)
```

## M3ProcessorConfig

### 功能说明
异常值抑制M3算法处理器配置类，用于配置M3异常值抑制处理器相关的参数。

### 类原型
```python
class M3ProcessorConfig(BaseModel):
    pass
```

### 调用示例
```python
from msmodelslim.quant.session.session import M3ProcessorConfig

m3_processor_cfg = M3ProcessorConfig()
```

## M4ProcessorConfig

### 功能说明
异常值抑制M4算法处理器配置类，用于配置M4异常值抑制处理器相关的参数。

### 类原型
```python
class M4ProcessorConfig(BaseModel):
    pass
```

### 调用示例
```python
from msmodelslim.quant.session.session import M4ProcessorConfig

m4_processor_cfg = M4ProcessorConfig()
```

## M6Config

### 功能说明
异常值抑制M6算法参数配置类，用于配置M6异常值抑制相关的参数。

### 类原型
```python
class M6Config(BaseModel):
    alpha: float = None
    beta: float = None
```

### 参数说明
| 参数名| 输入/返回值 | 含义 | 使用限制 |
| ------ | ------ | ------ | ------ |
| alpha | 输入 | 控制算法的平滑程度 | 可选。<br>数据类型：float。取值范围为 [0, 1]，默认值 None。如果 alpha 和 beta 均指定为具体数值，则直接使用这些值；如果任一值未传入，算法将自动进行寻优以计算最优的 alpha 和 beta 值，用于异常值抑制。 |
| beta | 输入 | 控制算法的平滑程度 | 可选。<br>数据类型：float。取值范围为 [0, 1]，默认值 None。如果 alpha 和 beta 均指定为具体数值，则直接使用这些值；如果任一值未传入，算法将自动进行寻优以计算最优的 alpha 和 beta 值，用于异常值抑制。 |


### 调用示例
```python
from msmodelslim.quant.session.session import M6Config

m6_cfg = M6Config(alpha=0.8, beta=0.2)
```

## M6ProcessorConfig

### 功能说明
异常值抑制M6算法处理器配置类，用于配置M6量化处理器相关的参数。

### 类原型
```python
class M6ProcessorConfig(BaseModel):
    cfg: M6Config
```

### 参数说明
| 参数名| 输入/返回值 | 含义 | 使用限制 |
| ------ | ------ | ------ | ------ |
| cfg | 输入 | M6异常值抑制配置 | 必选。<br>数据类型：M6Config，M6异常值抑制配置类。|


### 调用示例
```python
from msmodelslim.quant.session.session import M6ProcessorConfig, M6Config

m6_processor_cfg = M6ProcessorConfig(
    cfg=M6Config(
        alpha=0.8,
        beta=0.2
    )
)
```
