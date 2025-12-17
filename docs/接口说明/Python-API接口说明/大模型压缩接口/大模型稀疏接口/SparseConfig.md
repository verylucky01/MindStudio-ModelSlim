## SparseConfig

### 功能说明
权重稀疏的参数配置类，保存权重稀疏过程中配置的参数。

### 函数原型
```python
SparseConfig(mode='sparse', method='magnitude', sparse_ratio=0.5, progressive=False, uniform=True)
```

### 参数说明
| 参数名| 输入/返回值 | 含义 | 使用限制 |
| ------ | ------ | ------ | ------ |
| mode | 输入 | 将工具配置为压缩模式。| 可选。<br>数据类型：String。<br>默认为'sparse', 可选值为['sparse']。 |
| method | 输入 | 配置具体的稀疏算法类型。| 可选。<br>数据类型：String。<br>默认为'magnitude'，可选值为['magnitude', 'hessian', 'par', 'par_v2']。 |
| sparse_ratio | 输入 | 配置权重稀疏率。| 可选。<br>数据类型：float。<br>默认为0.5，可选范围为(0, 1)。 |
| progressive | 输入 | 配置渐进式稀疏模式。| 可选。<br>数据类型：bool。<br>默认为False。|
| uniform | 输入 | 配置全局自适应稀疏模式。| 可选。<br>数据类型：bool。<br>默认为True。<br>True：开启均匀稀疏。False：非均匀稀疏。|

### 调用示例
```python
from msmodelslim.pytorch.sparse.sparse_tools import SparseConfig
sparse_config = SparseConfig(method='magnitude', sparse_ratio=0.5)
```