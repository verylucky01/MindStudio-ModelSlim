## quant_model()

### 功能说明
在多模态生成模型量化中，需要调用统一的量化接口进行量化，此函数用于根据量化会话配置调用量化核心逻辑完成量化。

### 函数原型
```python
quant_model(model: nn.Module, session_cfg: SessionConfig)
```

### 参数说明
| 参数名 | 输入/返回值 | 含义 | 使用限制 |
| ------ | ---------- | ---- | -------- |
| model | 输入 | 多模态生成模型需要量化的部分。 | 必选。<br>数据类型：nn.Module，当前仅支持对多模态生成模型的transformer部分进行量化，加载完整pipeline后，选择pipeline.transformer作为model。|
| session_cfg | 输入 | 量化会话配置类，用于配置量化相关的参数、校准数据以及运行设备。| 必选。<br>数据类型：SessionConfig。|

### 调用示例

```python
import torch
from ascend_utils.common.security.pytorch import safe_torch_load
from msmodelslim.quant.session.session import W8A8ProcessorConfig, W8A8QuantConfig, SaveProcessorConfig
from msmodelslim.quant.session.session import SessionConfig, quant_model

session_config = SessionConfig(
    processor_cfg_map={
        "w8a8": W8A8ProcessorConfig(
            cfg=W8A8QuantConfig(
                act_method='minmax'
            ),
            disable_names=[]
        ),
        "save": SaveProcessorConfig(
            output_path="./",
            safetensors_name=None,
            json_name=None,
            save_type=['safe_tensor'],
            part_file_size=None
        )
    },
    calib_data=safe_torch_load("calib_data.pth"),
    device="npu"
)

# 加载pipeline
pipeline = load_pipeline(...)

model = pipeline.transformer

quant_model(model, session_config)
```