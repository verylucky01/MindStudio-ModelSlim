# -*- coding: utf-8 -*-
# Copyright Huawei Technologies Co., Ltd. 2023-2023. All rights reserved.
__all__ = ['QuantConfig']

from msmodelslim.onnx.squant_ptq.quant_config import QuantConfig

try:
    from msmodelslim.onnx.squant_ptq.onnx_quant_tools import OnnxCalibrator
except ModuleNotFoundError as exception:
    from msmodelslim import logger
    logger.warning("Can not import OnnxCalibrator from: %s", exception)
else:
    __all__ += ['OnnxCalibrator']

