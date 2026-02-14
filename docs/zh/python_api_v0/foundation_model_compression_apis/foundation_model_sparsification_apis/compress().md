## compress()

### 功能说明
运行权重稀疏算法，初始化Compressor之后，通过compress()函数来执行权重稀疏。

### 函数原型
```python
prune_compressor.compress(dataset)
```

### 参数说明
| 参数名| 输入/返回值 | 含义 | 使用限制 |
| ------ | ------ | ------ | ------ |
| dataset | 输入 | 稀疏校准数据集。| 必选。<br>数据类型：list。 |

### 调用示例
```python
import torch
import torch_npu
from msmodelslim.pytorch.sparse.sparse_tools import SparseConfig, Compressor
sparse_config = SparseConfig(method='magnitude', sparse_ratio=0.5)
# model 是一个pytorch定义的nn.Module模型，以一个简单神经网络模型为例
class TwoLayerNet(torch.nn.Module):
    def __init__(self, D_in, H, D_out):
        super(TwoLayerNet, self).__init__()
        self.linear1 = torch.nn.Linear(D_in, H, bias=True)
        self.linear2 = torch.nn.Linear(H, D_out, bias=True)
    def forward(self, x):
        x = self.linear1(x)
        y_pred = self.linear2(x)
        return y_pred
D_in, H, D_out = 100, 10, 1
model = TwoLayerNet(D_in, H, D_out)
prune_compressor = Compressor(model, sparse_config)
test_dataset = [torch.randn(64, D_in)]
prune_compressor.compress(dataset=test_dataset)
```