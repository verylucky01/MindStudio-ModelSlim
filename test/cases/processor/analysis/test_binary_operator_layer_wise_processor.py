#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""
-------------------------------------------------------------------------
This file is part of the MindStudio project.
Copyright (c) 2026 Huawei Technologies Co.,Ltd.

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
from types import SimpleNamespace
from unittest.mock import MagicMock, call, patch

import torch
import torch.nn as nn

from msmodelslim.core.base.protocol import BatchProcessRequest
from msmodelslim.processor.analysis.binary_operator_layer_wise.processor import BinaryOperatorLayerWiseProcessor
from msmodelslim.utils.exception import UnexpectedError


class TinyDecoderLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(4, 4)
        self.linear2 = nn.Linear(4, 4)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        return x


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([TinyDecoderLayer()])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class TestLayerAnalysisProcessor(unittest.TestCase):
    """测试 LayerAnalysisProcessor（mse_layer_wise 指标）。"""

    def setUp(self):
        self.model = TinyModel()
        self.adapter = MagicMock()
        self.config = SimpleNamespace(
            metrics="mse_layer_wise",
            patterns=["layers.0"],
            configs=[MagicMock(name="cfg1")],
        )
        self.request = BatchProcessRequest(
            name="layers.0",
            module=self.model.layers[0],
            datas=[torch.randn(1, 4)],
        )

    def _build_fake_method(self):
        fake_method = MagicMock()
        fake_method.name = "mse_layer_wise"
        fake_method.get_hook.return_value = (
            lambda module, input_tensor, output_tensor, layer_name, stats_dict: None
        )
        return fake_method

    @patch("msmodelslim.processor.analysis.binary_operator_layer_wise.processor.AutoSessionProcessor.from_config")
    @patch("msmodelslim.processor.analysis.binary_operator_layer_wise.processor.LayerWiseMethodFactory.create_method")
    def test_init_set_empty_state_when_config_valid(self, mock_create_method, mock_from_config):
        fake_method = self._build_fake_method()
        qp = MagicMock()
        mock_create_method.return_value = fake_method
        mock_from_config.return_value = qp

        processor = BinaryOperatorLayerWiseProcessor(self.model, self.config, adapter=self.adapter)

        self.assertEqual(processor.config, self.config)
        self.assertIs(processor.adapter, self.adapter)
        self.assertEqual(processor.quant_processors, [qp])
        mock_from_config.assert_has_calls([
            call(self.model, self.config.configs[0], self.adapter),
        ])
        mock_create_method.assert_called_once_with("mse_layer_wise", adapter=self.adapter)
        self.assertEqual(processor._target_decoder_layers, [])
        self.assertEqual(processor._float_layer_stats, {})
        self.assertEqual(processor._quant_layer_stats, {})
        self.assertEqual(processor._decoder_layer_scores, [])
        self.assertEqual(processor._hook_handles, {})

    @patch("msmodelslim.processor.analysis.binary_operator_layer_wise.processor.get_current_context")
    @patch("msmodelslim.processor.analysis.binary_operator_layer_wise.processor.AutoSessionProcessor.from_config")
    @patch("msmodelslim.processor.analysis.binary_operator_layer_wise.processor.LayerWiseMethodFactory.create_method")
    def test_pre_run_call_quant_processors_when_context_exists(
        self, mock_create_method, mock_from_config, mock_get_current_context
    ):
        fake_method = self._build_fake_method()
        qp = MagicMock()
        mock_create_method.return_value = fake_method
        mock_from_config.return_value = qp
        mock_get_current_context.return_value = {"layer_analysis": SimpleNamespace(debug={})}

        processor = BinaryOperatorLayerWiseProcessor(self.model, self.config, adapter=self.adapter)
        processor.pre_run()

        qp.pre_run.assert_called_once()

    @patch("msmodelslim.processor.analysis.binary_operator_layer_wise.processor.get_current_context")
    @patch("msmodelslim.processor.analysis.binary_operator_layer_wise.processor.AutoSessionProcessor.from_config")
    @patch("msmodelslim.processor.analysis.binary_operator_layer_wise.processor.LayerWiseMethodFactory.create_method")
    def test_pre_run_raise_unexpected_error_when_context_missing(
        self, mock_create_method, mock_from_config, mock_get_current_context
    ):
        fake_method = self._build_fake_method()
        mock_create_method.return_value = fake_method
        mock_from_config.return_value = MagicMock()
        mock_get_current_context.return_value = None

        processor = BinaryOperatorLayerWiseProcessor(self.model, self.config, adapter=self.adapter)

        with self.assertRaises(UnexpectedError):
            processor.pre_run()

    @patch("msmodelslim.processor.analysis.binary_operator_layer_wise.processor.AutoSessionProcessor.process")
    @patch("msmodelslim.processor.analysis.binary_operator_layer_wise.processor.AutoSessionProcessor.from_config")
    @patch("msmodelslim.processor.analysis.binary_operator_layer_wise.processor.LayerWiseMethodFactory.create_method")
    def test_preprocess_collect_float_stats_and_clear_hooks(
        self, mock_create_method, mock_from_config, mock_super_process
    ):
        fake_method = self._build_fake_method()
        fake_method.filter_layers_by_patterns.return_value = ["layers.0"]
        # get_linear_layers_for_decoder 返回两个 Linear 子模块
        fake_method.get_linear_layers_for_decoder.return_value = [
            "layers.0.linear1",
            "layers.0.linear2",
        ]
        mock_create_method.return_value = fake_method
        mock_from_config.return_value = MagicMock()

        processor = BinaryOperatorLayerWiseProcessor(self.model, self.config, adapter=self.adapter)
        # 为了简单，直接把 model 传入 request.module 的上层名字中
        request = BatchProcessRequest(name="layers.0", module=self.model, datas=[torch.randn(1, 4)])

        processor.preprocess(request)

        fake_method.filter_layers_by_patterns.assert_called_once_with(
            ["layers.0"],
            self.config.patterns,
        )
        fake_method.get_linear_layers_for_decoder.assert_called()
        mock_super_process.assert_called_once_with(request)
        # 预处理结束后，不应残留 hook
        self.assertEqual(processor._hook_handles, {})

    @patch("msmodelslim.processor.analysis.binary_operator_layer_wise.processor.AutoSessionProcessor.from_config")
    @patch("msmodelslim.processor.analysis.binary_operator_layer_wise.processor.LayerWiseMethodFactory.create_method")
    def test_process_call_quant_processors_in_order_when_quant_processors_exist(
        self, mock_create_method, mock_from_config
    ):
        fake_method = self._build_fake_method()
        qp = MagicMock()
        mock_create_method.return_value = fake_method
        mock_from_config.return_value = qp

        processor = BinaryOperatorLayerWiseProcessor(self.model, self.config, adapter=self.adapter)
        processor.process(self.request)

        qp.preprocess.assert_called_once_with(self.request)
        qp.process.assert_called_once_with(self.request)
        qp.postprocess.assert_called_once_with(self.request)

    @patch("msmodelslim.processor.analysis.binary_operator_layer_wise.processor.AutoSessionProcessor.process")
    @patch("msmodelslim.processor.analysis.binary_operator_layer_wise.processor.AutoSessionProcessor.from_config")
    @patch("msmodelslim.processor.analysis.binary_operator_layer_wise.processor.LayerWiseMethodFactory.create_method")
    def test_postprocess_compute_decoder_scores_and_clear_stats(
        self, mock_create_method, mock_from_config, mock_super_process
    ):
        fake_method = self._build_fake_method()
        fake_method.compute_decoder_layer_scores.return_value = [
            {"name": "layers.0", "score": 0.5},
        ]
        fake_method.filter_layers_by_patterns.return_value = ["layers.0"]
        fake_method.get_linear_layers_for_decoder.return_value = [
            "layers.0.linear1",
            "layers.0.linear2",
        ]

        mock_create_method.return_value = fake_method
        mock_from_config.return_value = MagicMock()

        processor = BinaryOperatorLayerWiseProcessor(self.model, self.config, adapter=self.adapter)
        processor._target_decoder_layers = ["layers.0"]
        processor._float_layer_stats = {
            "layers.0.linear1": {"output": torch.randn(1, 4)},
            "layers.0.linear2": {"output": torch.randn(1, 4)},
        }
        processor._quant_layer_stats = {
            "layers.0.linear1": {"output": torch.randn(1, 4)},
            "layers.0.linear2": {"output": torch.randn(1, 4)},
        }

        request = BatchProcessRequest(name="layers.0", module=self.model, datas=[torch.randn(1, 4)])

        processor.postprocess(request)

        mock_super_process.assert_called_once_with(request)
        fake_method.compute_decoder_layer_scores.assert_called_once()
        self.assertEqual(processor._decoder_layer_scores, [{"name": "layers.0", "score": 0.5}])
        # 处理完当前块后，应清理对应的统计
        self.assertEqual(processor._float_layer_stats, {})
        self.assertEqual(processor._quant_layer_stats, {})

    @patch("msmodelslim.processor.analysis.binary_operator_layer_wise.processor.get_current_context")
    @patch("msmodelslim.processor.analysis.binary_operator_layer_wise.processor.AutoSessionProcessor.from_config")
    @patch("msmodelslim.processor.analysis.binary_operator_layer_wise.processor.LayerWiseMethodFactory.create_method")
    def test_post_run_call_quant_processors_and_set_context_when_scores_ready(
        self, mock_create_method, mock_from_config, mock_get_current_context
    ):
        fake_method = self._build_fake_method()
        fake_method.name = "mse_layer_wise"
        qp = MagicMock()
        mock_create_method.return_value = fake_method
        mock_from_config.return_value = qp

        processor = BinaryOperatorLayerWiseProcessor(self.model, self.config, adapter=self.adapter)
        processor._decoder_layer_scores = [{"name": "layers.0", "score": 1.0}]
        fake_ctx = {"layer_analysis": SimpleNamespace(debug={})}
        mock_get_current_context.return_value = fake_ctx

        processor.post_run()

        qp.post_run.assert_called_once()
        self.assertEqual(fake_ctx["layer_analysis"].debug["layer_scores"], processor._decoder_layer_scores)
        self.assertEqual(fake_ctx["layer_analysis"].debug["method"], "mse_layer_wise")
        self.assertEqual(fake_ctx["layer_analysis"].debug["patterns"], self.config.patterns)


if __name__ == "__main__":
    unittest.main()

