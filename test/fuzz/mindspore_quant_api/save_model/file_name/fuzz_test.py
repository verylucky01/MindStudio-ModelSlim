#!/usr/bin/python3
# Copyright 2022 Huawei Technologies Co., Ltd

"""
Usage:
TEST_PATH=$PWD/test/fuzz/mindspore_quant_api/save_model/file_name
python3 -m coverage run ${TEST_PATH}/fuzz_test.py ${TEST_PATH}/samples/ -atheris_runs=1000
"""

import sys
import logging

import atheris
import numpy as np


@atheris.instrument_func
def fuzz_test(input_bytes):
    # Put mindspore related imports inside, avoiding ARM atheris `cannot allocate memory in static TLS block` error.
    import mindspore as ms
    ms.set_context(device_target='CPU')  # NPU will be rather slow

    from test.resources import sample_net_mindspore as sample_net
    from msmodelslim.mindspore.quant.ptq_quant.create_config import create_quant_config
    from msmodelslim.mindspore.quant.ptq_quant.quantize_model import quantize_model
    from msmodelslim.mindspore.quant.ptq_quant.save_model import save_model

    file_name = input_bytes.decode('utf-8', 'ignore').strip()
    logging.info("file_name: %s", file_name)

    input_height, input_width = np.random.uniform(32, 224, size=(2,)).astype('int32')
    input_data = ms.Tensor(np.random.uniform(size=[1, 3, input_height, input_width]), dtype=ms.float32)

    model_list = [
        sample_net.SampleModel,
        sample_net.TestNetMindSpore,
        sample_net.TestNetMindSpore2,
        sample_net.TestNetMindSpore4,
    ]
    model = np.random.choice(model_list)()
    file_format = np.random.choice(['AIR', 'MINDIR', 'ONNX'])
    logging.info("model: %s, file_format: %s", model.__class__.__name__, file_format)

    config_file = "mindspore_quant_api_save_model_fuzz_test_config_file.json"
    create_quant_config(config_file, model)
    quantized_model = quantize_model(config_file, model, input_data)

    try:
        save_model(file_name, quantized_model, input_data, file_format=file_format)
    except ValueError as value_error:
        logging.error(value_error)


if __name__ == '__main__':
    import os

    TEST_SAVE_PATH = "automl_fuzz_test_save_path"
    os.makedirs(TEST_SAVE_PATH, exist_ok=True)
    os.chdir(TEST_SAVE_PATH)

    atheris.Setup(sys.argv, fuzz_test)
    atheris.Fuzz()
