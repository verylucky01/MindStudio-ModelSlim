## 模型低秩分解

深度学习运算，尤其是CV（计算机视觉）和NLP（自然语言处理）类任务运算，包含大量的矩阵运算，而低秩分解通过将大矩阵分解为若干个低秩矩阵的乘积，从而降低存储空间和计算量，降低推理开销。

当前支持在训练服务器上对MindSpore和PyTorch框架下模型进行低秩分解，执行前需参考[安装指南](../../../install_guide.md)完成开发环境配置。

### 操作步骤
1. 用户需自行准备模型、训练脚本和数据集，本样例以PyTorch框架的ResNet50和数据集ImageNet为例。

2. 编辑模型的训练脚本pytorch_resnet50_apex.py文件，导入低秩分解的库文件。
```python
from msmodelslim.pytorch import low_rank_decompose
# 请根据实际情况导入对应框架的库文件
from ascend_utils.common.utils import count_parameters 
from ascend_utils.common.security import SafeWriteUmask
```
- 说明
MindSpore框架下库文件的路径为msmodelslim.mindspore.low_rank_decompose，PyTorch框架下库文件的路径为 msmodelslim.pytorch.low_rank_decompose。

3. （可选）调整日志输出等级，启动训练任务后，将日志输出等级设置为相应级别，以便在训练时显示对应的日志信息。[日志级别说明](../../../python_api/common_apis.md#参数说明)
```
from msmodelslim import set_logger_level
set_logger_level("info")        #根据实际情况配置
```
4. （可选）在模型创建后，使用count_parameters接口，配置日志打屏显示内容。启动调优任务后，将打屏显示原模型的参数量信息。请参考count_parameters配置。
```
print("Original model parameters:", count_parameters(model))
```

5. 在模型创建后，使用Decompose类接口，配置低秩分解的方式，请参考Decompose进行配置。
```
decomposer = low_rank_decompose.Decompose(model).from_ratio(0.5) 
```

6. 使用Decompose类的decompose_network接口，实际执行低秩分解，并返回分解后的模型，请参考decompose_network进行配置。
```
model = decomposer.decompose_network()    #使用分解后的模型替换原模型
```

7. （可选）配置日志打屏显示内容，启动调优任务后，将打屏显示分解后模型的参数量信息。
```
print("Decomposed model parameters:", count_parameters(model))
```

8. 多卡训练时，需要先在单卡训练下保存模型权重（单卡训练时无需执行本步骤，直接启动调优任务即可）。
```
state_dict_file = "/home/xxx/decompose_state_dict.ckpt"   #请根据实际情况配置模型权重文件保存的路径及名称
with SafeWriteUmask():
    torch.save(model.state_dict(),state_dict_file)
```

9. 多卡训练时，需要在多卡下指定 do_decompose_weight=False，只转换模型结构为低秩分解后的模型，不分解模型权重。然后加载单卡训练下保存的权重（单卡训练时无需执行本步骤，直接启动调优任务即可）。
```
from ascend_utils.common.security.pytorch import safe_torch_load
model = decomposer.decompose_network(do_decompose_weight=False)
model.load_state_dict(state_dict = safe_torch_load(state_dict_file, map_location="cpu"))
```

10. 启动训练任务，根据单卡或多卡调用不同的执行脚本，并指定data_path为数据集路径。
- 单卡训练时，执行命令启动训练任务。
```
bash ./test/train_full_1p.sh --data_path=./datasets/imagenet  #请根据实际情况配置数据集路径
```
- 多卡训练时，执行命令启动训练任务，会在步骤8指定路径下生成模型权重文件。以下示例为8卡训练，请根据实际情况替换启动脚本。
```
bash ./test/train_full_8p.sh --data_path=./datasets/imagenet   #请根据实际情况配置数据集路径
```

11. 查看结果。
训练完成后输出模型训练精度和性能信息。