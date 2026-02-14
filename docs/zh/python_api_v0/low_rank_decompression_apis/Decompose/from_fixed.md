## from_fixed

### 功能说明 
Decompose类方法，模型低秩分解各层的分解率配置，以固定值的方式指定各层分解后中间层的channel数，返回自身用于链式调用。

### 函数原型
```python
from_fixed(channel_fixed, excludes=None, divisor=64)
```

### 参数说明
| 参数名| 输入/返回值 | 含义 | 使用限制 |
| ------ | ------ | ------ | ------ |
| channel_fixed | 输入 | 各层的固定分解channel数。| 必选。<br>数据类型：整数，且大于0。 |
| excludes | 输入 | 指定不分解的层名称。| 可选。<br>数据类型：None或列表或元组。<br>默认值为None。 |
| divisor | 输入 | 指定分解后channel的倍率，如指定16，则分解后的channel数为16的倍数。| 可选。<br>数据类型：int，需大于0，默认值为64。 <br>说明：divisor设置为1时，表示禁用此功能。|

### 调用示例
```python
from msmodelslim.pytorch import low_rank_decompose
decomposer = low_rank_decompose.Decompose(model)  # 调用__init__初始化类
decomposer = decomposer.from_fixed(64,divisor=16)  # 按照fixed方式计算分解信息
```