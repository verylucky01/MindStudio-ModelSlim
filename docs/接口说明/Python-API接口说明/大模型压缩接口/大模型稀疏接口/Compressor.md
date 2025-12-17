## Compressor

### 功能说明
权重稀疏参数配置类，通过Compressor类封装稀疏算法。

### 函数原型
```python
Compressor(model, cfg)
```

### 参数说明
| 参数名| 输入/返回值 | 含义 | 使用限制 |
| ------ | ------ | ------ | ------ |
| model | 输入 | 需要进行权重稀疏的模型。| 必选。<br>数据类型：nn.Module。 |
| cfg | 输入 | 已配置的SparseConfig类。| 必选。<br>数据类型：SparseConfig。 |

### 调用示例
```python
from msmodelslim.pytorch.sparse.sparse_tools import SparseConfig, Compressor
sparse_config = SparseConfig(method='magnitude', sparse_ratio=0.5)
prune_compressor = Compressor(model, sparse_config)   # model 是一个pytorch定义的nn.Module模型
```