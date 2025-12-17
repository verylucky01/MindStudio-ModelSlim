# 训练后量化（PyTorch）


## 概述
训练后量化工具需要用户提供PyTorch训练脚本或者pth文件，工具可自动对模型中的卷积和线性层（torch.nn.Linear和torch.nn.Conv2d）进行识别并量化，最终导出量化后的onnx模型，量化后的模型可以在推理服务器上运行，达到提升推理性能的目的。量化过程中用户需自行提供模型与数据集，调用API接口完成模型的量化调优。

## 自动混合精度量化算法
为了提升量化精度，训练后量化（PyTorch）算法内置了自动混合精度的模块，自动识别并回退量化敏感层为浮点计算，避免量化敏感层对精度造成较大损失。算法核心是：计算每个量化层量化前后输出的MSE，根据MSE的排序来衡量每一个量化层的量化敏感性，自动回退MSE最大的部分敏感层，从而提升量化的精度。

## 精度保持策略
为了进一步降低量化精度损失，训练后量化（PyTorch）工具内集成了多种精度保持策略，对权重的量化参数和取整方式进行优化。
- Easy Quant权重优化方法：利用输出相似性优化量化参数，减少输入输出张量的量化误差，推荐在data-free模式下使用，通常能够起到较好的改善效果。
- ADMM权重优化方法：使用交替优化的方法，对权重的量化参数进行迭代更新优化，推荐在label-free模式下使用，适当改善量化效果。
- Rounding取整优化：在量化中普通取整不是最优解，使用自适应取整的方式优化权重的取整能提高量化精度，推荐在label-free模式下使用，适当改善量化效果。

## 调用示例

```python
import torchvision

from msmodelslim.pytorch.quant.ptq_tools import QuantConfig, Calibrator

from ascend_utils.common.security import SafeWriteUmask

if __name__ == '__main__':
    MODEL_ARCH = "resnet50"
    SAVE_PATH = "./output"
    INPUTS_NAMES = ["input.1"]

    model = torchvision.models.resnet50(pretrained=True)
    model.eval()

    disable_names = []
    input_shape = [1, 3, 224, 224]
    keep_acc = {'admm': [False, 1000], 'easy_quant': [False, 1000], 'round_opt': False}

    quant_config = QuantConfig(
        disable_names=disable_names,  # 手动回退的量化层名称，要求格式list[str]，如精度太差，推荐回退量化敏感层，如分类层、输入层、检测head层等
        amp_num=0,  # 混合精度量化回退层数，要求格式int；默认为0
        input_shape=input_shape,  # 模型输入的shape，用于data-free量化构造虚拟数据
        keep_acc=keep_acc,  # 精度保持策略
        sigma=25,  # 大于0使用sigma统计方法；传入0值使用min-max统计方法。
    )

    calibrator = Calibrator(model, quant_config)

    calibrator.run()  # 执行量化算法

    calibrator.export_quant_onnx("resnet50", "./output", ["input.1"])  # 用来导出昇腾可部署的量化onnx模型

```
## 多模态量化场景
说明
多模态量化，当前硬件限定 Atlas 800I A2 / 800T A2 / 900 A2, 当前量化已经支持但不仅限于SD3和opensora1.2。
多模态量化场景导入代码样例：
```python
import torch
from diffusers import StableDiffusion3Pipeline

from ascend_utils.common.security.pytorch import safe_torch_load
from msmodelslim.pytorch.quant.ptq_tools import Calibrator, QuantConfig


pipe = StableDiffusion3Pipeline.from_pretrained(
    "/stable-diffusion-3-medium-diffusers/", 
    torch_dtype=torch.float16, 
    local_files_only=True
    ).to("npu") #模型路径
pipe.set_progress_bar_config(disable=True)
base = pipe
model = pipe.transformer

calib_dataset = safe_torch_load("sd3_calib_data_v3.pth", map_location="npu")
quant_config = QuantConfig(
    w_bit=8,
    a_bit=8,
    w_signed=True,
    a_signed=True,
    w_sym=True,
    a_sym=False,
    act_quant=True,
    act_method=1,
    quant_mode=1,
    disable_names=None,
    amp_num=0,
    keep_acc=None,
    sigma=25,
    device="npu" #设置模型运行device
)
calibrator = Calibrator(model, quant_config, calib_dataset)
calibrator.run()
calibrator.export_quant_safetensor("/output_path/")
```

### 校准数据获取方式
在上面的示例中，校准数据为 sd3_calib_data_v3.pth，其获取方式如下：
加载 SD3 预训练模型 --> 添加 Listener 类用于捕捉模型输入参数 --> 配置 calib_prompts --> 遍历 calib_prompts，输入 Listener 类中执行前向推理（num_inference_steps用于配置一个prompt生成多少个数据）--> 保存校准数据

代码参考如下：
```python
import torch
from diffusers import StableDiffusion3Pipeline

calib_data = []

class Listener(torch.nn.Module):
    def __init__(self, module):
        super(Listener, self).__init__()
        self.module = module
        self.inputs = []
    
    def forward(self, *args, **kwargs):
        sample = {}
        for k in kwargs:
            if isinstance(kwargs[k], torch.Tensor):
                sample[k] = kwargs[k].cpu()
            else:
                sample[k] = kwargs[k]
        self.inputs.append(sample)
        return self.module(*args, **kwargs)

pipe = StableDiffusion3Pipeline.from_pretrained(
    "path_to_stable-diffusion-3-medium-diffusers", 
    torch_dtype=torch.float16, 
    local_files_only=True
    )
pipe.to("npu")  # 使用gpu时则为pipe.to("cuda")

pipe.transformer = Listener(pipe.transformer)

# 校准prompts，根据需要配置多条
calib_prompts = ['a photo of a cat holding a sign that says hello world']

for prompt in calib_prompts:
    image = pipe(
        prompt=prompt,
        negative_prompt="",
        num_inference_steps=28,  # 此处配置为28，则一个prompt将会生成28条校准数据
        height=1024,
        width=1024,
        guidance_scale=7.0,
    ).images[0]

calib_data = pipe.transformer.inputs

with SafeWriteUmask(umask=0o377):
    torch.save(calib_data, "path_to_save/sd3_calib_data.pth")
```
