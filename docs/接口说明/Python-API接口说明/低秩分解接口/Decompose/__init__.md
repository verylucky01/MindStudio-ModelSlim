## init

### 功能说明 
Decompose类方法，对用户输入的模型进行类初始化。

### 函数原型
```python
__init__(model, config_file=None)
```

### 参数说明
| 参数名| 输入/返回值 | 含义 | 使用限制 |
| ------ | ------ | ------ | ------ |
| model | 输入 | 待低秩分解模型。| 必选。<br>数据类型：PyTorch或MindSpore模型。 |
| config_file | 输入 | 分解后各层中间层channel信息保存的文件路径和文件名称。| 可选。<br>数据类型：字符串或None，默认值为None，表示不保存分解后信息，可指定 json 结尾的字符串，在调用 from_xxx()接口后，保存channel信息到该文件中。 |


### 调用示例
```python
from msmodelslim.pytorch import low_rank_decompose
decomposer = low_rank_decompose.Decompose(model)  # 调用__init__初始化类
```