# Copyright Huawei Technologies Co., Ltd. 2021-2022. All rights reserved.


def replace_module(network, name, module, backend="mindspore"):
    tokens = name.split('.')
    sub_tokens = tokens[:-1]
    cur_network = network
    for token in sub_tokens:
        if not hasattr(cur_network, token):
            return
        cur_network = getattr(cur_network, token)
    setattr(cur_network, tokens[-1], module)
    if backend == "mindspore":
        module.update_parameters_name(name + '.')
    if tokens[-1].isdigit():
        idx = int(tokens[-1])
        cur_network[idx] = module
