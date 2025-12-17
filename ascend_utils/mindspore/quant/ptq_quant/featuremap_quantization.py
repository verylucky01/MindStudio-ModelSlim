# Copyright Huawei Technologies Co., Ltd. 2021-2022. All rights reserved.

import numpy as np
import mindspore.nn as nn
from mindspore import Tensor

from ascend_utils.mindspore.quant.ptq_quant.simulated_quant import SimulatedQuant


def rename_parameters(cell, prefix):
    """rename the parameters of cell with input prefix"""
    for key in cell.parameters_dict().keys():
        sufix = key.split('.')[-1]
        new_name = '.'.join([prefix, sufix])
        cell.parameters_dict()[key].name = new_name


def insert_child_to_cell(network, target_cell_name, new_subcell):
    """ insert the new_subcell to original's father cell """
    change = False
    cell_name_list = target_cell_name.split('.')
    sub_cell = None
    for cell_name, sub_cell in network.cells_and_names():
        if cell_name == '.'.join(cell_name_list[:-1]):
            sub_cell.insert_child_to_cell(cell_name_list[-1], new_subcell)
            change = True
            break
    if isinstance(sub_cell, nn.SequentialCell) and change:
        sub_cell.cell_list = list(sub_cell.cells())


def quant_activation(network, cell_name, cell):
    num_bins = 128
    max_percentile = 0.999999
    min_percentile = 0.999999
    search_range_start = 0.7
    search_range_end = 1.3
    search_step = 0.01
    shape = (1,)
    scale_init = Tensor(np.ones(*shape).astype(np.float32))
    offset_init = Tensor(np.zeros(*shape).astype(np.float32))
    new_subcell = SimulatedQuant(
        cell,
        cell_name,
        scale_init,
        offset_init,
        num_bins,
        min_percentile,
        max_percentile,
        search_range_start,
        search_range_end,
        search_step,
        True
    )
    insert_child_to_cell(network, cell_name, new_subcell)