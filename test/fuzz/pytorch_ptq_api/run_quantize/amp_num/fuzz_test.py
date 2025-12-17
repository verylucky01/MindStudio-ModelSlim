#!/usr/bin/python3
# Copyright 2022 Huawei Technologies Co., Ltd

"""
Usage:
TEST_PATH=test/fuzz/onnx_ptq_api/run_quantize/file_name
python3 -m coverage run ${TEST_PATH}/fuzz_test.py ${TEST_PATH}/samples/ -atheris_runs=1000
"""

import sys
import os

import atheris
import numpy as np
import torch

with atheris.instrument_imports():
    from resources.sample_net_torch import TestOnnxQuantModel
    from msmodelslim import logger
    from msmodelslim.pytorch.quant.ptq_tools import QuantConfig
    try:
        from msmodelslim.pytorch.quant.ptq_tools import Calibrator
    except ImportError as ee:
        logger.error("can not import Calibrator, skip")
        Calibrator = None

ONNX_MODEL_PATH = "./test.onnx"


@atheris.instrument_func
def fuzz_test(input_bytes):
    fdp = atheris.FuzzedDataProvider(input_bytes)
    amp_num = fdp.ConsumeIntInRange(0, 100)

    logger.info("amp_num: %d", amp_num)

    input_shape = [1, 3, 32, 32]
    quant_config = QuantConfig(disable_names=[], amp_num=amp_num, input_shape=input_shape)
    if Calibrator is None:
        return  # Skip

    model = TestOnnxQuantModel()
    try:
        calibrator = Calibrator(model, quant_config)
    except ValueError as value_error:
        logger.error(value_error)
        return
    calibrator.run()


if __name__ == '__main__':
    TEST_SAVE_PATH = "automl_fuzz_test_save_path"
    os.makedirs(TEST_SAVE_PATH, exist_ok=True)
    os.chdir(TEST_SAVE_PATH)
    atheris.Setup(sys.argv, fuzz_test)
    atheris.Fuzz()
    if os.path.exists(ONNX_MODEL_PATH):
        os.remove(ONNX_MODEL_PATH)
    if os.path.exists(TEST_SAVE_PATH):
        os.removedirs(TEST_SAVE_PATH)
