# Copyright Huawei Technologies Co., Ltd. 2022-2022. All rights reserved.

from resources.sample_net_mindspore import TestNetMindSpore

import mindspore as ms 
import numpy as np

from ascend_utils.common.security.mindspore import check_mindspore_cell, check_mindspore_input


def test_check_mindspore_module_given_valid_when_any_then_pass():
    model = TestNetMindSpore(class_num=10)
    check_mindspore_cell(model)


def test_check_mindspore_input_given_valid_when_any_then_pass():
    data = np.random.randn(3, 4).astype(np.float32)
    tensor = [ms.Tensor(data, dtype=ms.float32)]
    check_mindspore_input(tensor)