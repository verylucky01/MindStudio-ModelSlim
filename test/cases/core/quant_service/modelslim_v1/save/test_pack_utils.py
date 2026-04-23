#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""
-------------------------------------------------------------------------
This file is part of the MindStudio project.
Copyright (c) 2025 Huawei Technologies Co.,Ltd.

MindStudio is licensed under Mulan PSL v2.
You can use this software according to the terms and conditions of the Mulan PSL v2.
You may obtain a copy of Mulan PSL v2 at:

         http://license.coscl.org.cn/MulanPSL2

THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
See the Mulan PSL v2 for more details.
-------------------------------------------------------------------------
"""

import unittest

import torch

from msmodelslim.core.quant_service.modelslim_v1.save.utils.pack import pack_fp4_to_uint8


class TestPackUtils(unittest.TestCase):
    def test_pack_fp4_to_uint8_exact_value_and_sign_mapping(self):
        x = torch.tensor([[0.0, 0.5, -1.0, -6.0]], dtype=torch.float32)

        packed = pack_fp4_to_uint8(x)

        expected = torch.tensor([[16, 250]], dtype=torch.uint8)
        self.assertEqual(packed.dtype, torch.uint8)
        self.assertEqual(packed.tolist(), expected.tolist())

    def test_pack_fp4_to_uint8_nearest_value_mapping(self):
        # 0.75 / 1.25 are tie cases and should pick lower index due to argmin behavior.
        x = torch.tensor([[0.75, 1.25, 2.6, -3.4]], dtype=torch.float32)

        packed = pack_fp4_to_uint8(x)

        expected = torch.tensor([[33, 213]], dtype=torch.uint8)
        self.assertEqual(packed.tolist(), expected.tolist())

    def test_pack_fp4_to_uint8_multi_row_shape_and_values(self):
        x = torch.tensor(
            [
                [4.2, -0.4, 1.49, -1.51],
                [-0.1, 6.1, -2.2, 2.9],
            ],
            dtype=torch.float32,
        )

        packed = pack_fp4_to_uint8(x)

        expected = torch.tensor(
            [
                [150, 179],
                [120, 92],
            ],
            dtype=torch.uint8,
        )
        self.assertEqual(tuple(packed.shape), (2, 2))
        self.assertEqual(packed.tolist(), expected.tolist())

    def test_pack_fp4_to_uint8_odd_n_raises_runtime_error(self):
        x = torch.tensor([[0.0, 0.5, 1.0]], dtype=torch.float32)

        with self.assertRaises(RuntimeError):
            pack_fp4_to_uint8(x)
