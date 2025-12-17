# Copyright (c) 2023-2024 Huawei Technologies Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import stat

import unittest
from resources.sample_net_torch import TestOnnxQuantModel
import torch
import pytest

from msmodelslim.pytorch.quant import ptq_tools
from msmodelslim.pytorch.quant.ptq_tools import QuantConfig
from msmodelslim.pytorch.quant.ptq_tools import Calibrator

@pytest.mark.skipif("Calibrator" not in ptq_tools.__all__, reason="requires KIA so")
class TestQuantizeGivenTorchModel(unittest.TestCase):
    def test_run_quantize_given_conv2d_model_when_datafree_then_pass(self):
        model = TestOnnxQuantModel()
        onnx_model_path = "./test.onnx"
        input_data = torch.randn((1, 3, 32, 32))
        torch.onnx.export(model,
                          input_data,
                          onnx_model_path,
                          input_names=['input'],
                          output_names=['output'])
        os.chmod(onnx_model_path, stat.S_IRUSR | stat.S_IWUSR)

        disable_names = []
        input_shape = [1, 3, 32, 32]
        quant_config = QuantConfig(disable_names=disable_names,
                                   amp_num=1,
                                   input_shape=input_shape)
        
        calibrator = Calibrator(model, quant_config)
        calibrator.run()

        input_names = ["input.1"]
        output_path = "./"
        output_model_name = "TestQuantModel"
        calibrator.export_quant_onnx(output_model_name,
                                     output_path,
                                     input_names,
                                     save_fp=True)
        
        self.assertTrue(os.path.exists("./" + "{}_quant.onnx".format(output_model_name)))
        self.assertTrue(os.path.exists("./" + "{}_fp.onnx".format(output_model_name)))

        os.remove("./" + "{}_quant.onnx".format(output_model_name))
        os.remove("./" + "{}_fp.onnx".format(output_model_name))
        os.remove(onnx_model_path)
