## 训练后量化（MindSpore）

### 概述
训练后量化过程中，用户需自行提供训练好的权重文件和一小批验证集用来矫正量化因子，调用API接口完成模型的调优过程。目前支持MindSpore框架模型的量化调优。模型量化期间，用户可手动配置参数，并使用部分数据完成对模型的校准，获取一个量化后的模型。

执行训练后量化前，需参考[安装指南](../../../install_guide.md)完成开发环境配置。

### 操作步骤
1. 用户自行准备预训练模型和数据集。本样例以ResNet50模型为例，获取模型结构定义脚本，并参考[README](https://gitee.com/mindspore/models/blob/master/official/cv/ResNet/README_CN.md)下载所需数据集， 以cifar10数据集为例，在config/resnet50_cifar10_config.yaml里配置data_path和checkpoint_file_path，并在eval.py的基础上进行修改。

2. 新建模型量化脚本resnet50_quant.py，将eval.py内容复制到该文件中，删除eval_net()函数中定义损失，定义metric，计算metric相关的代码，保留如下初始化模型和加载权重相关的代码。

需要根据实际情况修改的配置项包括：
- **config.data_path**：数据集路径
- **config.batch_size**：批次大小
- **config.eval_image_size**：评估图像尺寸
- **config.class_num**：分类数量
- **config.checkpoint_file_path**：预训练权重文件路径

```python
target = config.device_target
# init context
ms.set_context(mode=ms.GRAPH_MODE, device_target=target, save_graphs=False)
if target == "Ascend":
    device_id = int(os.getenv('DEVICE_ID'))
    ms.set_context(device_id=device_id)
# create dataset
dataset = create_dataset(dataset_path=config.data_path, do_train=False, batch_size=config.batch_size,
                            eval_image_size=config.eval_image_size,
                            target=target)
# define net
model = resnet(class_num=config.class_num)
# load checkpoint
param_dict = ms.load_checkpoint(config.checkpoint_file_path)
ms.load_param_into_net(model, param_dict)
model.set_train(False)
```

3. 在resnet50_quant.py中导入量化接口
```
from msmodelslim.mindspore.quant.ptq_quant.quantize_model import quantize_model
from msmodelslim.mindspore.quant.ptq_quant.create_config import create_quant_config
from msmodelslim.mindspore.quant.ptq_quant.save_model import save_model
```

4. （可选）调整日志输出等级，启动调优任务后，将打印显示量化调优的日志信息。
```
import logging
logging.getLogger().setLevel(logging.INFO)    #请根据实际情况进行配置
```

5. 准备预训练模型。
当前脚本中已存在加载预训练模型的相关代码，可跳过此步骤，其他模型请参考以下内容进行配置。
```
model = get_user_network()   #加载模型结构，返回模型实例
load_checkpoint(ckpt_file_path, model)    #模型加载预训练参数，请根据实际情况配置
```

6. 使用create_quant_config接口生成配置文件。
```
config_file = "./quant_config.json"    #待生成的量化配置文件存放路径及名称，请根据实际情况配置
create_quant_config(config_file, model)
```

7. 使用quantize_model接口修改原模型，进行量化算子插入，此处的input_data需要与预训练模型的input保持一致的shape。
```
import mindspore as ms
from mindspore import dtype as mstype
input_data = ms.Tensor(np.random.uniform(size=[256, 3, 224, 224]), dtype=mstype.float32)    #请根据实际情况配置
model_calibrate = quantize_model(config_file, model, input_data)    #通过调用quantize_model接口生成的量化后的模型
```

8. 对量化模型进行校准。校准过程中会使用少量数据集进行前向传播，以校准量化算子中的参数，提高量化后的精度。
```
for i, data in enumerate(dataset.create_dict_iterator(num_epochs=1)):
    model(data['image'])
    if i >= 2:
        break
```

9. 使用save_model接口保存量化后模型。
```
file_name = "./quantized_model"    #指定量化后模型保存路径和文件名
save_model(file_name, model_calibrate, input_data, file_format="AIR")    #请根据需要配置量化后模型的格式
```

10. 启动模型量化调优任务，并在步骤9指定的目录获取一个量化完成的模型。
```
python3 resnet50_quant.py
```
