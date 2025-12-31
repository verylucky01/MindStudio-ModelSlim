# -*- coding: utf-8 -*-
# Copyright Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.

import pytest

from msmodelslim.pytorch.llm_ptq.llm_ptq_tools.quant_config.quant_config_classes import KVQuantConfig
from msmodelslim.pytorch.llm_ptq.llm_ptq_tools.quant_config.quant_config_classes import BaseConfig
from ..old_quant_config import OldQuantConfig
from ..compare_old_and_new_config import \
    compare_config_parameters


class TestKVQuantConfig:
    def test_kv_quant_config_should_equal_to_old_quant_config_when_default_value(self):
        base_config = BaseConfig()
        kv_quant_config = KVQuantConfig(base_config)
        old_quant_config = OldQuantConfig(use_kvcache_quant=True)
        assert compare_config_parameters(kv_quant_config, old_quant_config) is True
