## count_parameters

### 功能说明
统计模型参数量接口，根据用户提供的模型，统计模型参数量。

### 函数原型
```python
count_parameters(network)
```

### 参数说明
| 参数名| 输入/返回值 | 含义 | 使用限制 |
| ------ | ------ | ------ | ------ |
| network | 输入 | 待低秩分解模型。| 必选。<br>数据类型：PyTorch或MindSpore模型。 |


### 调用示例
```python
from ascend_utils.common.utils import count_parameters
print("Original model parameters:", count_parameters(network))
```