# -*- coding: utf-8 -*-
# Copyright Huawei Technologies Co., Ltd. 2020-2021. All rights reserved.
__all__ = ['set_logger_level']

from msmodelslim.utils.logging import logger, set_logger_level
from msmodelslim.utils.patch import patch_torch, patch_pydantic

patch_torch()
patch_pydantic()

OLD_PACKAGE_NAME = 'modelslim'
NEW_PACKAGE_NAME = 'msmodelslim'
