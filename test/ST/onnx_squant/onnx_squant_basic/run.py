# -*- coding: utf-8 -*-
# Copyright Huawei Technologies Co., Ltd. 2023-2023. All rights reserved.
import cv2
import numpy as np
import os
from modelslim.onnx.squant_ptq import OnnxCalibrator, QuantConfig

def get_calib_data():
    img = cv2.imread(f"{os.environ['PROJECT_PATH']}/resource/onnx_squant/000000000139.jpg")
    img_data = cv2.resize(img, (224, 224))
    img_data = img_data[:, :, ::-1].transpose(2, 0, 1).astype(np.float32)
    img_data /= 255.
    img_data = np.expand_dims(img_data, axis=0)
    return [[img_data]]

input_model = f"{os.environ['PROJECT_PATH']}/resource/onnx_squant/resnet50_official_1batch.onnx"
output_path = f"{os.environ['PROJECT_PATH']}/output/onnx_squant/resnet50_quant.onnx"


disable_names = []
config = QuantConfig(disable_names=disable_names,
                     quant_mode=1,
                     amp_num=0,
                     disable_first_layer=False,
                     disable_last_layer=False)

calib_data = get_calib_data()
calib = OnnxCalibrator(input_model, config)
calib.run()
calib.export_quant_onnx(output_path)
del calib