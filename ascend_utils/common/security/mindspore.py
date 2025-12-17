# Copyright Huawei Technologies Co., Ltd. 2022-2022. All rights reserved.

import mindspore as ms

MAX_DEPTH_THRESHOLD = 100


def check_mindspore_cell(cell):
    if not isinstance(cell, ms.nn.Cell):
        raise TypeError("model must be a Mindspore.nn.Cell instance. Not {}".format(type(cell)))


def check_mindspore_input(input_data):
    """
    Use recursion to check whether the input_data is Mindspore.Tensor

    Args:
        input_data: can be list/tuple/Tensor
    """
    if not input_data or len(input_data) == 0:
        raise ValueError("input data cannot be empty")

    def recursive_check_mindspore_input(cur_data, depth=0):
        if depth >= MAX_DEPTH_THRESHOLD:
            raise ValueError("input data nested too deeply")
        depth = depth + 1
        if isinstance(cur_data, (list, tuple)):
            for value in cur_data:
                recursive_check_mindspore_input(value, depth)
        elif not isinstance(cur_data, ms.Tensor):
            raise TypeError("input data must be Mindspore.Tensor. Not {}".format(type(cur_data)))

    recursive_check_mindspore_input(input_data)
