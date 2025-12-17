#!/usr/bin/python3
# Copyright 2022 Huawei Technologies Co., Ltd

"""
Usage:
TEST_PATH=$PWD/test/fuzz/low_rank_decompose/from_dict/channel_dict/
python3 -m coverage run ${TEST_PATH}/fuzz_test.py ${TEST_PATH}/samples/ -atheris_runs=1000
"""

import sys
import logging
import json

import atheris

with atheris.instrument_imports():
    from msmodelslim.pytorch.low_rank_decompose import Decompose
    from resources.sample_net_torch import LrdSampleNetwork

@atheris.instrument_func
def fuzz_test(input_bytes):
    channel_dict = input_bytes.decode('utf-8', 'ignore').strip()
    logging.info("channel_dict: %s", channel_dict)

    model = LrdSampleNetwork()
    decomposer = Decompose(model, "low_rank_decompose_test.json")

    try:
        channel_dict = json.loads(channel_dict)
    except json.JSONDecodeError:
        return

    try:
        decomposer.from_dict(channel_dict)
    except ValueError as value_error:
        logging.error(value_error)
        return

    decomposer.from_file()


if __name__ == '__main__':
    import os

    TEST_SAVE_PATH = "automl_fuzz_test_save_path"
    os.makedirs(TEST_SAVE_PATH, exist_ok=True)
    os.chdir(TEST_SAVE_PATH)

    atheris.Setup(sys.argv, fuzz_test)
    atheris.Fuzz()
