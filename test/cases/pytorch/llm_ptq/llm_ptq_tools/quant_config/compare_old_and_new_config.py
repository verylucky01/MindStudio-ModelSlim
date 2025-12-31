# -*- coding: utf-8 -*-
# Copyright Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.

def compare_config_parameters(old_config, new_config):
    old_param = old_config.__dict__
    new_param = new_config.__dict__
    common_keys = set(old_param.keys()) & set(new_param.keys())
    for key in common_keys:
        if key == '_cur_config':
            continue
        if old_param[key] != new_param[key]:
            return False
    return True
