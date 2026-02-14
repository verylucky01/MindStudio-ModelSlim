## QuantConfig

### 功能说明
量化参数配置类，保存量化过程中配置的参数。

### 函数原型
```python
QuantConfig(disable_names=None, fraction=0.01)
```

### 参数说明
| 参数名| 输入/返回值 | 含义 | 使用限制 |
| ------ | ------ | ------ | ------ |
| disable_names | 输入 | 需排除量化的节点名称，即手动回退的量化层名称。<br>如精度太差，推荐回退量化敏感层，如分类层、输入层、检测head层等。| 可选。<br>数据类型：object。|
| fraction | 输入 | 稀疏量化精度控制。|  可选。<br>数据类型：float。<br>取值范围[0.01,0.1]。 |

### 调用示例
```python
from msmodelslim.mindspore.llm_ptq import Calibrator, QuantConfig
quant_config = QuantConfig(disable_names=["lm_head"], fraction=0.04)
```