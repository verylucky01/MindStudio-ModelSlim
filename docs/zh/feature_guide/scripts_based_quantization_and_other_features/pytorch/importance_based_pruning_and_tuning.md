## 基于重要性评估的剪枝调优

msModelSlim工具提供了基于重要性评估的模型剪枝调优API，用户只需要提供模型实例，即可调用剪枝API完成模型的剪枝。剪枝后的模型提升了一定的性能，减少了模型的大小，提升推理过程中的效率。

目前支持PyTorch框架下的模型剪枝调优，执行剪枝调优前，请先参考[安装指南](../../../install_guide.md)完成开发环境配置。

- 注意：该功能仅支持 PyTorch 2.0.0 以上版本。

### 操作步骤

1. 用户自行准备待剪枝模型实例以及训练脚本。本样例以torchvision中的vgg16为例。

2. 打开待剪枝模型的训练脚本vision/references/classification/train.py。编辑train.py文件，导入剪枝接口。剪枝API接口说明请参考PruneTorch。
```
from msmodelslim.pytorch.prune.prune_torch import PruneTorch
```

3. （可选）调整日志输出等级，启动调优任务后，将打印显示设置级别的日志信息。[日志级别说明](../../../python_api/common_apis.md#参数说明)
```
from msmodelslim import set_logger_level
set_logger_level("info")        #根据实际情况配置
```

4. 在原脚本初始化网络，并已经加载权重后，使用PruneTorch接口自定义配置剪枝的重要性评估函数、算子节点保留的参数比例、剪枝率等。
```
desc = PruneTorch(model, torch.ones([1, 3, 224, 224]).type(torch.float32)).prune(0.8)
```

5. 启动模型剪枝调优任务。建议使用原始训练过程最终学习率，训练10epoch即可。
```
python3 train.py --model vgg16 --lr 1e-5 --epochs 10 --pretrained --batch-size 256 -j 48
```
将得到一个剪枝后的模型，可进行后续的训练任务。


6. 在后续的评估过程中，参考如下示例配置，加载步骤4返回模型剪枝信息。
```
PruneTorch(model, torch.ones([1, 3, 224, 224]).type(torch.float32)).prune_by_desc(desc)
```