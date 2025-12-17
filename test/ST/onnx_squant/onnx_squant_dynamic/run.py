# -*- coding: utf-8 -*-
# Copyright Huawei Technologies Co., Ltd. 2023-2023. All rights reserved.
import os
from modelslim.onnx.squant_ptq import OnnxCalibrator, QuantConfig

input_model = f"{os.environ['PROJECT_PATH']}/resource/onnx_squant/swin_tiny_patch4_window7_224_dynamic.onnx"
output_path = f"{os.environ['PROJECT_PATH']}/output/onnx_squant/swin_tiny_patch4_window7_224_dynamic_quant.onnx"


disable_names = []
config = QuantConfig(disable_names=disable_names,
                     quant_mode=1,
                     amp_num=0,
                     disable_first_layer=False,
                     disable_last_layer=False,
                     is_dynamic_shape = True,
                     input_shape = [[1,3,224,224]])


calib = OnnxCalibrator(input_model, config)
calib.run()
calib.export_quant_onnx(output_path)
del calib