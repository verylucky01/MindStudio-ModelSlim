#  Copyright (c) Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.

import pytest
import numpy as np

from msmodelslim.pytorch.weight_compression.compress_utils import (
    pseudo_sparse,
    round_up,
    transform_nd2nz
)


class TestCompressUtils:
    def test_pseudo_sparse(self):
        np.random.seed(2024) # 不能删除，因为调用的pseudo_sparse函数中有numpy的随机数设置
        arr = np.array([1, 1, 1, 1])
        ratio = 0.5
        res = pseudo_sparse(arr, ratio)
        expected = np.array([1, 1, 0, 0])
        assert np.array_equal(res, expected)

    def test_round_up(self):
        val, align = 10, 1
        expected = 10
        assert round_up(val, align) == expected

        val, align = 5, 0
        expected = 0
        assert round_up(val, align) == expected

    def test_transform_nd2nz(self):
        nd_arr = np.random.rand(1024, 1024)
        nz_arr = transform_nd2nz(nd_arr)
        assert nz_arr.shape == (32, 64, 16, 32)



