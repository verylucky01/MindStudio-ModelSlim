#!/usr/bin/python3
# Copyright 2022 Huawei Technologies Co., Ltd

"""
Usage:
python3 -m coverage run fuzz_test.py ../samples/ -atheris_runs=1000
"""

import os
import sys

import atheris
import torch
# Need to import these first, otherwise `atheris.instrument_imports` will take a long time
import sympy as _
from scipy.optimize import minimize_scalar as _


class MyConvTestNet(torch.nn.Module):
    def __init__(self, groups=1) -> None:
        super().__init__()
        self.features = torch.nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2, groups=groups)

    def forward(self, x_in: torch.Tensor) -> torch.Tensor:
        x_out = self.features(x_in)
        return x_out


with atheris.instrument_imports():
    from msmodelslim.pytorch.prune.prune_torch import PruneTorch
    from msmodelslim import logger


@atheris.instrument_func
def fuzz_test(input_bytes):
    fdp = atheris.FuzzedDataProvider(input_bytes)
    prune_ratio = fdp.ConsumeFloat()

    logger.info("prune_ratio: %f", prune_ratio)

    model = MyConvTestNet()
    prune = PruneTorch(model, torch.ones([1, 3, 224, 224]).type(torch.float32))
    try:
        prune.analysis(prune_ratio)
    except ValueError as value_error:
        logger.error(value_error)
    except TypeError as type_error:
        logger.error(type_error)


if __name__ == '__main__':
    TEST_SAVE_PATH = "automl_fuzz_test_save_path"
    os.makedirs(TEST_SAVE_PATH, exist_ok=True)
    os.chdir(TEST_SAVE_PATH)

    atheris.Setup(sys.argv, fuzz_test)
    atheris.Fuzz()
    if os.path.exists(TEST_SAVE_PATH):
        os.removedirs(TEST_SAVE_PATH)
