#  -*- coding: utf-8 -*-
#  Copyright (c) 2025-2025 Huawei Technologies Co., Ltd.
#  #
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#  #
#  http://www.apache.org/licenses/LICENSE-2.0
#  #
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

import unittest
from typing import Tuple, Optional
from unittest.mock import patch

import torch
import torch.nn as nn

from msmodelslim.core.base.protocol import ProcessRequest
from msmodelslim.model.common.layer_wise_forward import (
    TransformersForwardBreak,
    generated_decoder_layer_visit_func,
    transformers_generated_forward_func
)


class DummyDecoderLayer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) \
            -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        return (x, None)


class SimpleModel(nn.Module):
    """简单的测试模型，包含一个线性层和一个TransformerDecoder层"""

    def __init__(self):
        super().__init__()
        self.decoder = DummyDecoderLayer()

    def forward(self, x: torch.Tensor) \
            -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        x = self.decoder(x)
        return x


class TestTransformersGenerated(unittest.TestCase):
    """测试_TransformersForwardBreak异常类和相关的生成器函数"""

    def setUp(self):
        """测试前的准备工作"""
        self.model = SimpleModel()

    def test_transformers_forward_break(self):
        """测试_TransformersForwardBreak异常类"""
        # 创建异常实例
        exception = TransformersForwardBreak()

        # 验证异常类型
        self.assertIsInstance(exception, Exception)

    def test_transformers_generated_visit_func(self):
        """测试transformers_generated_visit_func函数"""
        # 获取生成器
        generator = generated_decoder_layer_visit_func(self.model)

        # 获取第一个请求
        request = next(generator)

        # 验证请求
        self.assertIsInstance(request, ProcessRequest)
        self.assertEqual(request.name, "decoder")
        self.assertEqual(request.module, self.model.decoder)

    def test_transformers_generated_visit_func_with_custom_blocks(self):
        """测试transformers_generated_visit_func函数，使用自定义的transformer_blocks"""
        # 创建自定义的transformer_blocks
        transformer_blocks = [(name, module)
                              for name, module in self.model.named_modules()
                              if "decoder" in module.__class__.__name__.lower()]

        # 获取生成器
        generator = generated_decoder_layer_visit_func(self.model, transformer_blocks)

        # 获取第一个请求
        request = next(generator)

        # 验证请求
        self.assertIsInstance(request, ProcessRequest)
        self.assertEqual(request.name, "decoder")
        self.assertEqual(request.module, self.model.decoder)

    @patch('torch.distributed.get_rank')
    @patch('torch.distributed.get_world_size')
    @patch('torch.distributed.barrier')
    def test_transformers_generated_forward_func(self, mock_barrier, mock_get_world_size, mock_get_rank):
        """测试_transformers_generated_forward_func函数"""
        # 设置模拟函数
        mock_get_rank.return_value = 0
        mock_get_world_size.return_value = 1

        # 创建输入数据
        input_data = torch.randn(5, 10)

        # 获取生成器
        generator = transformers_generated_forward_func(self.model, input_data)

        # 获取第一个请求
        request = next(generator)

        # 验证请求
        self.assertIsInstance(request, ProcessRequest)
        self.assertEqual(request.name, "decoder")
        self.assertEqual(request.module, self.model.decoder)
        self.assertTrue(torch.equal(request.args[0], input_data))


if __name__ == '__main__':
    unittest.main()
