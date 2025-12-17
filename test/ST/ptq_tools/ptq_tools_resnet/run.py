import os
import torchvision
from modelslim.pytorch.quant.ptq_tools import QuantConfig, Calibrator

if __name__ == '__main__':
    MODEL_ARCH = "resnet50"
    SAVE_PATH = f"{os.environ['PROJECT_PATH']}/output/ptq-tools"
    INPUTS_NAMES = ["input.1"]

    os.makedirs(SAVE_PATH, exist_ok=True)

    model = torchvision.models.resnet50(pretrained=True)
    model.eval()

    disable_names = []
    input_shape = [1, 3, 224, 224]
    keep_acc = {'admm': [False, 1000], 'easy_quant': [False, 1000], 'round_opt': False}

    quant_config = QuantConfig(
        disable_names=disable_names,  # 手动回退的量化层名称，要求格式list[str]，如精度太差，推荐回退量化敏感层，如分类层、输入层、检测head层等
        amp_num=0,                    # 混合精度量化回退层数，要求格式int；默认为0
        input_shape=input_shape,      # 模型输入的shape，用于data-free量化构造虚拟数据
        keep_acc=keep_acc,            # 精度保持策略
        sigma=25,                     # 大于0使用sigma统计方法；传入0值使用min-max统计方法。
    )

    calibrator = Calibrator(model, quant_config)

    calibrator.run()   # 执行量化算法

    calibrator.export_quant_onnx(MODEL_ARCH, SAVE_PATH, INPUTS_NAMES)  # 用来导出昇腾可部署的量化onnx模型