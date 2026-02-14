## from_ratio

### 功能说明 
Decompose类方法，模型低秩分解各层的分解率配置，按照比例计算各层分解后的channel数，返回自身用于链式调用。

### 函数原型
```python
from_ratio(channel_ratio, excludes=None, divisor=64)
```

### 参数说明
| 参数名| 输入/返回值 | 含义 | 使用限制 |
| ------ | ------ | ------ | ------ |
| channel_ratio | 输入 | 需要分解的层按照该值计算分解后的channel 数。| 必选。<br>数据类型：float。范围 0-1。|
| excludes | 输入 | 指定不分解的层名称。| 可选。<br>数据类型：None或列表或元组。<br>默认值为None。 |
| divisor | 输入 | 指定分解后channel的倍率，如指定16，则分解后的channel数为16的倍数。| 可选。<br>数据类型：整数，需大于0，默认值为64。 <br>说明：divisor设置为1时，表示禁用此功能。|

### 调用示例
```python
from msmodelslim.pytorch import low_rank_decompose
decomposer = low_rank_decompose.Decompose(model)  # 调用__init__初始化类
decomposer = decomposer.from_ratio(0.5, divisor=16)  # 按照ratio方式计算分解信息
```