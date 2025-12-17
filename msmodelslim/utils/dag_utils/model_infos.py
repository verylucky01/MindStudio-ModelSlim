# Copyright Huawei Technologies Co., Ltd. 2023-2023. All rights reserved.

from enum import Enum


class ModuleType(str, Enum):
    CONV2D = 'Conv2d'
    ADD = '__add__'
    IADD = '__iadd__'
    LINEAR = 'Linear'
    BATCHNORM = 'BatchNorm2d'
    CONCAT = 'cat'
    RELU = 'ReLU'
    GETITEM = '__getitem__'
    LAYERNORM = 'Layernorm'