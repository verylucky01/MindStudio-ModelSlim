## from_file

### 功能说明 
Decompose 类方法，如果调用过其他的 from_xxx() 方法，且类初始化时指定的config_file有效，则从保存的文件中加载分解信息。

### 函数原型
```python
from_file()
```

### 调用示例
```python
from msmodelslim.pytorch import low_rank_decompose
decomposer = low_rank_decompose.Decompose(model)  #调用 __init__ 初始化类
decomposer = decomposer.from_file()
```