## decompose_network

### 功能说明 
按照用户配置的各层分解率，执行低秩分解，并返回分解后的模型。该函数通过递归遍历并处理模型的子模块`named_children`，递归的深度取决于网络的深度（即模块嵌套的层数）。若模型结构异常复杂或存在极端嵌套（如自定义超深网络），可能触发Python的递归深度限制。

### 函数原型
```python
decompose_network(do_decompose_weight=True, datasets=None, max_iter=-1)
```

### 参数说明
| 参数名| 输入/返回值 | 含义 | 使用限制 |
| ------ | ------ | ------ | ------ |
| do_decompose_weight | 输入 | 指定是否执行权重分解，并设置为分解后的模型。| 可选。<br>数据类型：布尔值。<br>默认值为True，可取值为True或False，若取值为False，则只将模型转化为低秩分解后的模型结构，各层权重为随机初始化的值。|
| datasets | 输入 | 输入数据集。若取值不为None时，则需要使用data-aware输入数据感知分解。| 可选。<br>默认值为None。若不为None，需输入模型可以直接遍历执行的数据集，其元素为 dict或list或tuple。 |
| max_iter | 输入 | datasets不为None，指定data-aware分解时，从datasets中取数据的最大迭代数| 可选。<br>数据类型：整数。默认值为-1，-1表示使用整个数据集，> 0 的值则表示在数据集中迭代的最大值。|

### 调用示例
```python
from msmodelslim.pytorch import low_rank_decompose
decomposer = low_rank_decompose.Decompose(model).from_ratio(0.5, divisor=16)  
model = decomposer.decompose_network(do_decompose_weight=True)
```