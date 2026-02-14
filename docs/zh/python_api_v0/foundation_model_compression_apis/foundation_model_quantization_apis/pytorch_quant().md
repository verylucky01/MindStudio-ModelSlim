## quant()

### 功能说明
quant函数用于对模型中的 Q、K 和 V 张量进行量化处理，以优化模型的性能和效率。

### 函数原型
```python
quant(states_tensor: torch.Tensor, qkv: str)
```

### 参数说明
| 参数名 | 输入/返回值 | 含义 | 使用限制 |
| ------ | ---------- | ---- | -------- |
| states_tensor | 输入 | 待量化的张量，可以是 Q、K 或 V 张量。 | 必选。<br>数据类型：torch.Tensor。 |
| qkv | 输入 | 指定当前 states_tensor 代表的张量类型，可以是 "q"、"k" 或 "v"。 | 必选。<br>数据类型：str。 |

### 调用示例

```python
from msmodelslim.pytorch.llm_ptq.llm_ptq_tools.fa_quant import FAQuantizer 
from msmodelslim import logger 

self.fa_quantizer = FAQuantizer(self.config, logger)

# 注意：fa量化功能要求将q、k、v都输入一次，否则输入不完全会报错。
query_states = self.fa_quantizer.quant(query_states, qkv="q")
key_states = self.fa_quantizer.quant(key_states, qkv="k")
value_states = self.fa_quantizer.quant(value_states, qkv="v")
```