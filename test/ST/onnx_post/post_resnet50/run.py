# -*- coding: utf-8 -*-
# Copyright Huawei Technologies Co., Ltd. 2023-2023. All rights reserved.
import os
from modelslim.onnx.post_training_quant import QuantConfig, run_quantize # 导入post_training_quant量化
from modelslim.onnx.post_training_quant.label_free.preprocess_func import preprocess_func_imagenet

def custom_read_data():
    calib_data = preprocess_func_imagenet(f"{os.environ['PROJECT_PATH']}/resource/onnx_squant/")
    return calib_data

calib_data = custom_read_data()

quant_config = QuantConfig(calib_data = calib_data, amp_num = 5) 
input_model_path = f"{os.environ['PROJECT_PATH']}/resource/onnx_post/resnet50_official_1batch.onnx" # 配置待量化模型的输入路径，请根据实际路径配置
output_model_path = f"{os.environ['PROJECT_PATH']}/output/onnx_post/resnet50_official_quant.onnx" # 配置量化后模型的名称及输出路径，请根据实际路径
run_quantize(input_model_path,output_model_path,quant_config) # 使用 run_quantize接口执行量化，配置
