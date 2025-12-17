# Copyright Huawei Technologies Co., Ltd. 2022-2022. All rights reserved.

"""
Usage:
python3 -m coverage run $(pwd)/fuzz_test.py $(pwd)/samples/ -atheris_runs=100  # execute code
coverage report -i  # coverage rate
coverage html -d foo -i  # coverage rate + code execution in a html
"""

import sys
import logging
import os
import random

import atheris

from test.fuzz.common.utils import random_change_dict_value

from msmodelslim.common.prune.transformer_prune.prune_model import PruneConfig

@atheris.instrument_func
def fuzz_test(input_bytes):

    fuzz_value = input_bytes.decode('utf-8', 'ignore').strip()

    patten = "uniter\.encoder\.encoder\.blocks\.(\d+)\."
    layer_id_map = {0: 1, 1: 3, 2: 5, 3: 7, 4: 9, 5: 11}

    test_case_id = random.randint(0, 1)
    if test_case_id == 0:
        random_change_dict_value(layer_id_map, fuzz_value)
    if test_case_id == 1:
        patten = fuzz_value

    config = PruneConfig()
    try:
        config.add_blocks_params(patten, layer_id_map)
    except ValueError as value_error:
        logging.error(value_error)
    except TypeError as type_error:
        logging.error(type_error)


if __name__ == '__main__':
    TEST_SAVE_PATH = "automl_fuzz_test_save_path"
    os.makedirs(TEST_SAVE_PATH, exist_ok=True)
    os.chdir(TEST_SAVE_PATH)

    atheris.Setup(sys.argv, fuzz_test)
    atheris.Fuzz()
