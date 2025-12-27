# Sparse tool
## 简介
稀疏算法是一种优化深度神经网络的技术，可以将linear网络层中不必要的参数置为0，部署阶段借助昇腾芯片unzip单元在线解码能力，可获得更加轻量化的模型，以提高模型的推理速度和泛化能力。
## 功能介绍
用户需自行准备模型，模型是基于pytorch网络结构，本样例以线性层为例。
1. 使用SparseConfig接口，配置稀疏参数、稀疏方式，生成稀疏化算法配置
```python
sparse_config = SparseConfig(method = "magnitude", sparse_ratio = 0.5, progressive = False, uniform = True)
```
- method: 稀疏方式，可选值为：'magnitude','hessian','par','par_v2'，默认'magnitude'
- sparse_ratio: 0~1,用户可以自行设置稀疏率, 默认0.5
- progressive: 渐进式稀疏，默认False
- uniform: 均匀稀疏，默认True

2. 用户自行准备一个batch的数据集作为稀疏算法的校准数据
```python
   test_dataset = [torch.randn(64, 100)]
```

3. 模型稀疏调优任务

```python
import torch
from msmodelslim.pytorch.sparse.sparse_tools import SparseConfig, Compressor

# 定义一个简单的模型
class SimpleModel(torch.nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.linear1 = torch.nn.Linear(100, 50)
        self.linear2 = torch.nn.Linear(50, 10)
    
    def forward(self, x):
        x = self.linear1(x)
        x = self.linear2(x)
        return x

generate_model = SimpleModel()
test_dataset = [torch.randn(64, 100)]
sparse_config = SparseConfig(method="magnitude", sparse_ratio=0.5)
prune_compressor = Compressor(generate_model, sparse_config)
prune_compressor.compress(dataset=test_dataset)
```
# example

```python
import torch
import torch_npu
from msmodelslim.pytorch.sparse.sparse_tools import SparseConfig, Compressor


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
model = TwoLayerNet(100, 10, 1)
test_dataset = [torch.randn(64, 100)]
sparse_config = SparseConfig(method='magnitude')
prune_compressor = Compressor(model, sparse_config)
prune_compressor.compress(dataset=test_dataset)
```