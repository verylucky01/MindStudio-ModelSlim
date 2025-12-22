## FAQuantizer

### 功能说明
运行量化算法，对Q（Query）、K（Key）、V（Value）进行量化。

### 函数原型
```python
FAQuantizer(config, logger)
```

### 参数说明
| 参数名 | 输入/返回值 | 含义 | 使用限制 |
| ------ | ---------- | ---- | -------- |
| config | 输入 | 配置类实例，包含模型参数。 | 必选。<br>自带config.json的大语言模型传入的config<br>数据类型：PretrainedConfig<br>不自带config.json的多模态模型传入的config<br>数据类型：Object。必须包含 'num_attention_heads'、'hidden_size' 、'num_key_value_heads'这三个参数。 |
| logger | 输入 | 日志记录器，用于记录日志信息。 | 必选。<br>数据类型：Logger。 |

### 调用示例
大语言模型调用示例：
```python
from msmodelslim.pytorch.llm_ptq.llm_ptq_tools.fa_quant import FAQuantizer 
from msmodelslim import logger 

self.fa_quantizer = FAQuantizer(self.config, logger)

query_states = self.fa_quantizer.quant(query_states, qkv="q")
key_states = self.fa_quantizer.quant(key_states, qkv="k")
value_states = self.fa_quantizer.quant(value_states, qkv="v")
```

多模态模型调用示例：
```python
# 实例化FAQuantizer类
# --------------------fa3-----------------------------
from msmodelslim.pytorch.llm_ptq.llm_ptq_tools.fa_quant import FAQuantizer 
from msmodelslim import logger 
from types import SimpleNamespace

config_dict = {
    'num_attention_heads': self.heads_num, 
    'hidden_size': self.hidden_size,
    'num_key_value_heads': self.heads_num,
    }

config = SimpleNamespace(**config_dict)
self.fa_quantizer = FAQuantizer(config, logger=logger)
# --------------------fa3-----------------------------
...   
# --------------------fa3-----------------------------
# 通过调用FAQuantizer的quant函数，对Q、K、V矩阵进行量化
query = self.fa_quantizer.quant(query, qkv="q")
key = self.fa_quantizer.quant(key, qkv="k")
value = self.fa_quantizer.quant(value, qkv="v")
# --
```