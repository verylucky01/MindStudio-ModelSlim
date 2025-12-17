# -*- coding: utf-8 -*-
# Copyright Huawei Technologies Co., Ltd. 2021-2022. All rights reserved.

from ascend_utils.mindspore.quant.ptq_quant.deploy.deploy_utils import get_op_type
from ascend_utils.mindspore.quant.ptq_quant.deploy.conv2d import Conv2dDeployQuant
from ascend_utils.mindspore.quant.ptq_quant.deploy.conv1d import Conv1dDeployQuant
from ascend_utils.mindspore.quant.ptq_quant.deploy.linear import LinearDeployQuant
from ascend_utils.mindspore.quant.ptq_quant.deploy.dense import DenseDeployQuant


class DeployQuantFactory:

    @staticmethod
    def creat_deploy_quant_op(simulated_quant_cell):
        quant_op_type = get_op_type(simulated_quant_cell.compute_cell)
        if quant_op_type == "Conv2d":
            return Conv2dDeployQuant(simulated_quant_cell)
        elif quant_op_type == "Conv1d":
            return Conv1dDeployQuant(simulated_quant_cell)
        elif quant_op_type == "Dense":
            return DenseDeployQuant(simulated_quant_cell)
        else:
            return LinearDeployQuant(simulated_quant_cell)