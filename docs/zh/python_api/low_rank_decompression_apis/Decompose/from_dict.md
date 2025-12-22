## from_dict

### 功能说明 
Decompose类方法，模型低秩分解各层的分解率配置，以字典的方式指定各层的分解后channel 数，返回自身用于链式调用。

### 函数原型
```python
from_dict(channel_dict, excludes=None, divisor=64)
```

### 参数说明
| 参数名| 输入/返回值 | 含义 | 使用限制 |
| ------ | ------ | ------ | ------ |
| channel_dict | 输入 | 字典方式根据层名称指定具体的分解方式。| 必选。<br>数据类型：字典类型，key 值为层名或者匹配层名的正则表达式，取值 为int值、float值、"vbmf"，分别对应参照 from_fixed、from_ratio、from_vbmf。<br>说明:在使用复杂的正则表达式时，用户需保证正则表达式的安全性，规避ReDoS攻击的风险，否则会引起程序执行缓慢。|
| excludes | 输入 | 指定不分解的层名称。| 可选。<br>数据类型:None或列表或元组。<br>默认值为None。 |
| divisor | 输入 | 指定分解后channel的倍率，如指定16，则分解后的channel数为16的倍数。| 可选。<br>数据类型：int，需大于0，默认值为64。 <br>说明：divisor设置为1时，表示禁用此功能。|

### 调用示例
```python
from msmodelslim.pytorch import low_rank_decompose
decomposer = low_rank_decompose.Decompose(model)  #调用__init__初始化类
decomposer = decomposer.from_dict({'feature.0': (64, 64), 'inner': 192, 'classifier.0': 128}, divisor=16)  # 按照dict方式计算分解信息
```