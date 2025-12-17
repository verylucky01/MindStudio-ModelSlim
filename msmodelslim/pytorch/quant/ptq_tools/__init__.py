# -*- coding: utf-8 -*-
# Copyright Huawei Technologies Co., Ltd. 2022-2022. All rights reserved.

__all__ = ['QuantConfig']

from msmodelslim.pytorch.quant.ptq_tools.quant_config import QuantConfig


try:
    from msmodelslim.pytorch.quant.ptq_tools.quant_tools import Calibrator
except ModuleNotFoundError as exception:
    from msmodelslim import logger

    logger.info("can not import Calibrator")
else:
    __all__ += ['Calibrator']
