# -*- coding: utf-8 -*-
# Copyright Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.

import os
import sys
import shutil
import mindspore as ms

from msmodelslim.mindspore.llm_ptq import QuantConfig as SparseQuantConfig


def test_sparse_quant_config():
    fraction = 0.011
    disable_names = [
        "lm_head",
        "model.layers.1.feed_forward.w1", "model.layers.1.feed_forward.w11", "model.layers.1.feed_forward.w3",
        "model.layers.2.feed_forward.w1", "model.layers.2.feed_forward.w11", "model.layers.2.feed_forward.w3",
        "model.layers.3.feed_forward.w1", "model.layers.3.feed_forward.w11", "model.layers.3.feed_forward.w3"
    ]

    quant_config = SparseQuantConfig(disable_names=disable_names, fraction=fraction)

    assert isinstance(quant_config, SparseQuantConfig)
    assert quant_config.fraction == fraction
    assert len(quant_config.disable_names) == 10