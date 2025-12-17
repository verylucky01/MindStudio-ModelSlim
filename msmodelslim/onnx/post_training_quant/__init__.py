# -*- coding: utf-8 -*-
# Copyright Huawei Technologies Co., Ltd. 2022-2022. All rights reserved.
from msmodelslim.onnx.post_training_quant.config import QuantConfig
from msmodelslim.onnx.post_training_quant.quantize import run_quantize, convert_version
from msmodelslim.onnx.post_training_quant.label_free.preprocess_func import preprocess_func_imagenet
from msmodelslim.onnx.post_training_quant.label_free.preprocess_func import preprocess_func_coco
