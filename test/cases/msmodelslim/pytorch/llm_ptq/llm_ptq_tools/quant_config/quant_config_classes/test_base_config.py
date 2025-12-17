# -*- coding: utf-8 -*-
# Copyright Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.

import pytest

from msmodelslim.pytorch.llm_ptq.llm_ptq_tools.quant_config.quant_config_classes import BaseConfig
from ..old_quant_config import OldQuantConfig
from ..compare_old_and_new_config import \
    compare_config_parameters


class TestBaseConfig:
    def test_base_config_should_equal_to_old_quant_config_when_default_value(self):
        old_quant_config = OldQuantConfig()
        base_config = BaseConfig()
        assert compare_config_parameters(old_quant_config, base_config) is True
