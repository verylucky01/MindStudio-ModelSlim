# Copyright Huawei Technologies Co., Ltd. 2022-2022. All rights reserved.
import os

import pytest
import torch

from msmodelslim import logger
from msmodelslim.pytorch.quant.qat_tools import qsin_qat, QatConfig
from msmodelslim.pytorch.quant.qat_tools import save_qsin_qat_model


class OneModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.device = torch.device('cpu')
        self.l1 = torch.nn.Linear(8, 8, bias=False)

    def get(self, x, y):

        return {x: y}

    def forward(self, x):
        x = self.l1(x)

        return x


@pytest.mark.skip()
def test_qat_finetune():
    model = OneModel()
    quant_config = QatConfig(grad_scale=0.0001, ema=0.99)
    model = qsin_qat(model, quant_config, logger)


@pytest.mark.skip()
def test_qat_deploy():
    fake_ckpt = "fake_checkpoint_asym.pth.tar"
    model = OneModel()
    torch.save(model, fake_ckpt)
    model = torch.load(fake_ckpt)
    save_onnx_name = "fake_test_qat_deploy.onnx"
    dummy_input = torch.FloatTensor(8, 8).type(torch.float32)
    input_names = ['input1']
    os.chmod(fake_ckpt, 0o755)
    save_qsin_qat_model(model, save_onnx_name, dummy_input, fake_ckpt, input_names)

    if os.path.exists(fake_ckpt):
        os.remove(fake_ckpt)

    if os.path.exists(save_onnx_name):
        os.remove(save_onnx_name)