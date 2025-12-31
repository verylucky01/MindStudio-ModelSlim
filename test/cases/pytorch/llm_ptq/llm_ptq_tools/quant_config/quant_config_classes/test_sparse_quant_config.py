# -*- coding: utf-8 -*-
# Copyright Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.

import pytest

from msmodelslim.pytorch.llm_ptq.llm_ptq_tools.quant_config.quant_config_classes import SparseQuantConfig
from msmodelslim.pytorch.llm_ptq.llm_ptq_tools.quant_config.quant_config_classes import BaseConfig
from ..old_quant_config import OldQuantConfig
from ..compare_old_and_new_config import \
    compare_config_parameters


class TestSparseQuantConfig:
    def test_sparse_quant_config_should_equal_to_old_quant_config_when_default_value(self):
        base_config = BaseConfig()
        sparse_quant_config = SparseQuantConfig(base_config)
        old_quant_config = OldQuantConfig(co_sparse=True)
        assert compare_config_parameters(sparse_quant_config, old_quant_config) is True
