# Copyright Huawei Technologies Co., Ltd. 2022-2022. All rights reserved.

import mindspore.nn as nn
from mindspore import load_checkpoint, load_param_into_net

from ascend_utils.common.prune.transformer_prune.prune_utils_ms import PRUNE_STATE_DICT_FUNCS_MS
from ascend_utils.common.security import get_valid_read_path
from ascend_utils.common.security.mindspore import check_mindspore_cell
from msmodelslim.common.prune.transformer_prune.prune_model import PruneConfig


def prune_model_weight_ms(model: nn.Cell, config: PruneConfig, ckpt_file_path: str):
    check_mindspore_cell(model)
    PruneConfig.check_prune_config(config, target_steps=list(PRUNE_STATE_DICT_FUNCS_MS.keys()))
    ckpt_file_path = get_valid_read_path(path=ckpt_file_path, extensions="ckpt")

    parameter_dict = load_checkpoint(ckpt_file_path)
    for step_name in config.prune_state_dict_steps:
        parameter_dict = PRUNE_STATE_DICT_FUNCS_MS.get(step_name)(model, parameter_dict, config)
    load_param_into_net(net=model, parameter_dict=parameter_dict, strict_load=True)
