## from_vbmf

### 功能说明 
Decompose 类方法，模型低秩分解各层的分解率配置，指定VBMF可变贝叶斯矩阵分解秩搜索方式，自动计算分解后的channel 数，返回自身用于链式调用，通常适用于有预训练权重的模型。

### 函数原型
```python
from_vbmf(excludes=None, divisor=64)
```

### 参数说明
| 参数名| 输入/返回值 | 含义 | 使用限制 |
| ------ | ------ | ------ | ------ |
| excludes | 输入 | 指定不分解的层名称。|可选。<br>数据类型：None或列表或元组。<br>默认值为None。|
| divisor | 输入 | 指定分解后channel的倍率，如指定16，则分解后的channel 数为 16 的倍数。| 可选。<br>数据类型：整数，且大于0，默认值为64。<br>说明：divisor设置为1时，表示禁用此功能。|


### 调用示例
```python
from msmodelslim.pytorch import low_rank_decompose
decomposer = low_rank_decompose.Decompose(model)  #调用__init__初始化类
decomposer = decomposer.from_vbmf(divisor=16)  #按照vbmf方式计算分解信息
```