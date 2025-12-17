# Copyright Huawei Technologies Co., Ltd. 2022-2022. All rights reserved.

import pytest

import torch.nn as nn

from ascend_utils.common.security.pytorch import check_torch_module


def test_check_torch_module_given_valid_when_any_then_pass():
    model = nn.Linear(1, 1)
    check_torch_module(model)


def test_check_torch_module_given_invalid_when_any_then_type_error():
    with pytest.raises(TypeError):
        # TypeError: value must be nn.Module
        check_torch_module(123)