# -*- coding: utf-8 -*-
# Copyright Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.
from ascend_utils.common.security import check_number


class RARopeCompressConfig:
    """ 
    The configuration for compression.
    config = RACompressConfig(induction_head_ratio=0.14,
                              echo_head_ratio=0.01)
    """

    def __init__(self, induction_head_ratio=0.14, echo_head_ratio=0.01):
        """
        Args:
            induction_head_ratio:控制induction head的判定比例
            echo_head_ratio:控制echoing head的判定比例
        """
        self.induction_head_ratio = induction_head_ratio
        self.echo_head_ratio = echo_head_ratio
        self._check_params()

    def _check_params(self):
        check_number(self.induction_head_ratio, float, 0, 1, param_name="induction_head_ratio")
        check_number(self.echo_head_ratio, float, 0, 1, param_name="echo_head_ratio")
