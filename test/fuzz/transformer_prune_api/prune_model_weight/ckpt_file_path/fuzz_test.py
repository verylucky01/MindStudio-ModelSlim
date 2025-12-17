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
from random import choice

import atheris


@atheris.instrument_func
def fuzz_test(input_bytes):
    import resources.sample_net_prune as sample_net
    from msmodelslim.common.prune.transformer_prune.prune_model import PruneConfig
    from msmodelslim.common.prune.transformer_prune.prune_model import prune_model_weight

    fuzz_value = input_bytes.decode('utf-8', 'ignore').strip()

    patten = "uniter\.encoder\.encoder\.blocks\.(\d+)\."
    layer_id_map = {0: 1, 1: 3, 2: 5, 3: 7, 4: 9, 5: 11}

    config = PruneConfig()
    config.add_blocks_params(patten, layer_id_map)
    model_list = [
        sample_net.TorchPrunedModel,
        sample_net.TorchOriModel,
        sample_net.MsPrunedModel,
        sample_net.MsOriModel,
    ]
    model = choice(model_list)()
    try:
        prune_model_weight(model=model, config=config, weight_file_path=fuzz_value)
    except ValueError as value_error:
        logging.error(value_error)
    except TypeError as type_error:
        logging.error(type_error)
    except Exception as ee:
        if not isinstance(ee.args[-1], (TypeError, ValueError)):  # prune_model_weight raises base Exception
            logging.error(ee)


if __name__ == '__main__':
    TEST_SAVE_PATH = "automl_fuzz_test_save_path"
    os.makedirs(TEST_SAVE_PATH, exist_ok=True)
    os.chdir(TEST_SAVE_PATH)

    atheris.Setup(sys.argv, fuzz_test)
    atheris.Fuzz()
