# 量化感知训练

## 简介

量化感知训练会重新训练量化模型，从而减小模型大小，并且加快推理过程。当前支持对PyTorch框架的CNN类模型进行量化，并将量化后的模型保存为.onnx文件，量化过程中，需要用户自行提供模型与数据集，调用API接口完成模型的量化调优。

## 使用前准备

已参考[安装指南](../../../install_guide.md)完成开发环境配置。
量化感知训练前须执行命令安装依赖。
如下命令如果使用非root用户安装，需要在安装命令后加上--user，例如：pip3 install onnx --user。

## 功能介绍
### 操作步骤

1. 用户需自行准备模型、训练脚本和数据集，本样例以PyTorch框架的Resnet50和数据集ImageNet为例。

2. 编辑训练脚本pytorch_resnet50_apex.py文件，导入如下接口。

3. 在优化器初始化之前调用“qsin_qat”函数，将量化后模型替换为“qsin_qat”的输出模型。请参考QatConfig和qsin_qat进行配置。同时在训练代码中，需注意保存伪量化模型权重ckpt文件，在导出量化onnx时使用。
```
quant_config = QatConfig(grad_scale=0.001)
quant_logger = get_logger()
model = qsin_qat(model, quant_config, quant_logger).to(model.device)     #根据实际情况配置待量化模型实例、量化配置和量化输出日志，注意需把模型按照原训练流程部署在NPU设备
```

4. 调用原训练流程进行单卡训练，执行train_full_1p.sh启动单卡训练任务。
```
bash ./test/train_full_1p.sh --data_path=/datasets/imagenet  #请根据实际情况配置数据集路径
```

5. 导出量化后的onnx模型。在伪量化模型权重ckpt文件保存后，新建quant_deploy.py文件，添加如下代码，调用“save_qsin_qat_model”函数，请参考save_qsin_qat_model进行配置。
```
import argparse
import os
import torch
from ascend_utils.common.security.pytorch import safe_torch_load
import models.image_classification.resnet as nvmodels

# 初始化模型
parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('-b', '--batch-size', default=1, type=int,
                    metavar='N',
                    help='onnx bs')
parser.add_argument('--pretrained', default="./org_model_best.pth.tar", type=str,
                    help='use pre-trained model')
parser.add_argument('--quant_ckpt', default="./checkpoint_77.244_asym.pth.tar", type=str,
                    help='use pre-trained model')

args = parser.parse_args()

model = nvmodels.build_resnet("resnet50", "classic", is_training=False)
pretrained_dict = safe_torch_load(args.pretrained, map_location='cpu')["state_dict"]
model.load_state_dict(pretrained_dict, strict=False)
#保存量化后的onnx模型
from msmodelslim.pytorch.quant.qat_tools import save_qsin_qat_model
#根据实际情况配置导出后模型文件名（文件后缀需为.onnx）、输入的shape、伪量化模型权重和onnx的输入名称
save_onnx_name='./resnet50.onnx'
dummy_input = torch.ones([args.batch_size, 3, 224, 224]).type(torch.float32)
saved_ckpt = args.quant_ckpt
input_names=['input1']
save_qsin_qat_model(model, save_onnx_name, dummy_input, saved_ckpt, input_names)  
```

6. 执行量化脚本，获取量化后的onnx模型。
```
python3 quant_deploy.py
```
