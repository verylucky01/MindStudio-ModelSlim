# Copyright Huawei Technologies Co., Ltd. 2022-2022. All rights reserved.

import torch.nn as nn

from ascend_utils.common.prune.transformer_prune.prune_utils_torch import PRUNE_STATE_DICT_FUNCS_TORCH
from ascend_utils.common.prune.transformer_prune.prune_utils_torch import PruneUtilsTorch
from ascend_utils.common.security import get_valid_read_path
from ascend_utils.common.security.pytorch import check_torch_module
from msmodelslim.common.prune.transformer_prune.prune_model import PruneConfig


def prune_model_weight_torch(model: nn.Module, config: PruneConfig, weight_file_path: str):
    check_torch_module(model)
    PruneConfig.check_prune_config(config, target_steps=list(PRUNE_STATE_DICT_FUNCS_TORCH.keys()))
    weight_file_path = get_valid_read_path(path=weight_file_path, extensions=["pt", "pth", "pkl", "bin"])

    state_dict = PruneUtilsTorch.get_state_dict(weight_file_path)
    for step_name in config.prune_state_dict_steps:
        state_dict = PRUNE_STATE_DICT_FUNCS_TORCH.get(step_name)(model, state_dict, config)
    model.load_state_dict(state_dict, strict=False)
