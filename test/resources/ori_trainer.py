# -*- coding: utf-8 -*-
# Copyright Huawei Technologies Co., Ltd. 2021-2021. All rights reserved.

from vega.common import ClassFactory, ClassType, Config

def run_train(model=None, vega_config=None, callback=None):
    config = Config()
    if vega_config is not None:
        vega_config.merge_to_config(config)
        vega_config.merge_to_dict(config)

    if callback is not None:
        callback.init_trainer()
        callback.before_train()
        callback.before_epoch(1)
        callback.before_train_step(1)
        callback.after_train_step(1)
        callback.after_epoch(1)
        callback.after_train()


def run_eval(model=None, vega_config=None, callback=None):
    config = Config()
    if vega_config is not None:
        vega_config.merge_to_config(config)
        vega_config.merge_to_dict(config)

    if callback is not None:
        callback.before_valid()
        callback.before_valid_step(1)
        callback.after_valid_step(1)
        callback.after_valid()

    return config.get("back", 1)