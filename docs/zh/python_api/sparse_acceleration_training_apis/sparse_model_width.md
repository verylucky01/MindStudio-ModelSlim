## sparse_model_width

### 功能说明
稀疏训练配置参数接口，判断当前稀疏化阶段，扩增模型与权重，以及调用重置 optimizer 接口，将用户提供的模型转化为稀疏化训练模型。

### 函数原型
```python
sparse_model_width(model, optimizer, steps_per_epoch, epochs_each_stage)
```

### 参数说明
| 参数名| 输入/返回值 | 含义 | 使用限制 |
| ------ | ------ | ------ | ------ |
| model | 输入 | 初始化后的原始模型。<br>若原训练脚本中已使用**torch.nn.parallel.DistributedDataParallel**封装了模型，model需为非ddp模式的模型。| 必选。<br>数据类型：PyTorch 模型。 |
| optimizer | 输入 | 初始化后的优化器。| 必选。<br>数据类型：PyTorch优化器，torch.optim.Optimizer的实例。 |
| steps_per_epoch | 输入 | 每个epoch的迭代数量，用于判断当前stage。| 必选。<br>数据类型：int，需大于0。 |
| epochs_each_stage | 输入 | 每个稀疏化阶段的epoch数量。| 必选。<br>数据类型：list或者tuple。元素必须是int。<br>长度大于2，且其中元素除最后一个需要为大于0的int值，最后一个元素可以为-1。<br>说明：epochs_each_stage最后一个元素为-1时，表示第三个训练阶段将一直进行，达到总epoch数量后才会停止。 |



### 调用示例
```python
from msmodelslim.pytorch import sparse
model = sparse.sparse_model_width(model, optimizer, steps_per_epoch=100, epochs_each_stage=[1, 2, 1])
```