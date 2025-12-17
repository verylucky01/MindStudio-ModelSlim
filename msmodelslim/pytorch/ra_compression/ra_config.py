# -*- coding: utf-8 -*-
# Copyright Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.
from ascend_utils.common.security import check_int, check_number


class RACompressConfig:
    """ 
    The configuration for compression.
    config = RACompressConfig(theta=0.00001,alpha=100)
    """

    def __init__(self, theta=0.00001, alpha=100):
        """
        Args:
            theta:attention score贡献度,保证校准后模型推理精度
            alpha:校准偏置,用于保证泛化性,控制窗口大小
        """
        self.theta = theta
        self.alpha = alpha
        self._check_params()

    def _check_params(self):
        check_int(self.alpha, 0, 10000, param_name="alpha")
        check_number(self.theta, float, 0.00001, 0.001, param_name="theta")
