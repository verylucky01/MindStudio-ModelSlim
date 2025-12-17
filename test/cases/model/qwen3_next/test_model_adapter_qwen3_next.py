# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import torch.nn as nn

from msmodelslim.core.base.protocol import ProcessRequest
from msmodelslim.core.const import DeviceType
from msmodelslim.core.graph import AdapterConfig, MappingConfig


class TestQwen3NextModelAdapter(unittest.TestCase):

    def setUp(self):
        self.model_type = 'Qwen3-Next-80B-A3B-Instruct'
        self.model_path = Path('..')

    @patch.dict('sys.modules', {
        'transformers.models.qwen3_next.modeling_qwen3_next': MagicMock(),
    })
    def test_get_model_type(self):
        """测试get_model_type方法"""
        from msmodelslim.model.qwen3_next.model_adapter import Qwen3NextModelAdapter
        with patch('msmodelslim.model.qwen3_next.model_adapter.TransformersModel.__init__', return_value=None):
            adapter = Qwen3NextModelAdapter(
                model_type=self.model_type,
                model_path=self.model_path
            )
            adapter.model_type = self.model_type

            result = adapter.get_model_type()
            self.assertEqual(result, self.model_type)

    @patch.dict('sys.modules', {
        'transformers.models.qwen3_next.modeling_qwen3_next': MagicMock(),
    })
    def test_get_model_pedigree(self):
        """测试get_model_pedigree方法"""
        from msmodelslim.model.qwen3_next.model_adapter import Qwen3NextModelAdapter
        with patch('msmodelslim.model.qwen3_next.model_adapter.TransformersModel.__init__', return_value=None):
            adapter = Qwen3NextModelAdapter(
                model_type=self.model_type,
                model_path=self.model_path
            )

            result = adapter.get_model_pedigree()
            self.assertEqual(result, 'qwen3_next')

    @patch.dict('sys.modules', {
        'transformers.models.qwen3_next.modeling_qwen3_next': MagicMock(),
    })
    def test_load_model(self):
        """测试load_model方法"""
        from msmodelslim.model.qwen3_next.model_adapter import Qwen3NextModelAdapter
        with patch('msmodelslim.model.qwen3_next.model_adapter.TransformersModel.__init__', return_value=None):
            adapter = Qwen3NextModelAdapter(
                model_type=self.model_type,
                model_path=self.model_path
            )

            mock_model = nn.Linear(10, 10)
            adapter._load_model = MagicMock(return_value=mock_model)

            result = adapter.load_model(device=DeviceType.NPU)

            self.assertIs(result, mock_model)
            adapter._load_model.assert_called_once_with(DeviceType.NPU)

    @patch.dict('sys.modules', {
        'transformers.models.qwen3_next.modeling_qwen3_next': MagicMock(),
    })
    def test_handle_dataset(self):
        """测试handle_dataset方法"""
        from msmodelslim.model.qwen3_next.model_adapter import Qwen3NextModelAdapter
        with patch('msmodelslim.model.qwen3_next.model_adapter.TransformersModel.__init__', return_value=None):
            adapter = Qwen3NextModelAdapter(
                model_type=self.model_type,
                model_path=self.model_path
            )

            mock_dataset = ['data1', 'data2']
            adapter._get_tokenized_data = MagicMock(return_value=mock_dataset)

            result = adapter.handle_dataset(dataset='test_data', device=DeviceType.CPU)

            self.assertEqual(result, mock_dataset)
            adapter._get_tokenized_data.assert_called_once_with('test_data', DeviceType.CPU)

    @patch.dict('sys.modules', {
        'transformers.models.qwen3_next.modeling_qwen3_next': MagicMock(),
    })
    def test_handle_dataset_by_batch(self):
        """测试handle_dataset_by_batch方法"""
        from msmodelslim.model.qwen3_next.model_adapter import Qwen3NextModelAdapter
        with patch('msmodelslim.model.qwen3_next.model_adapter.TransformersModel.__init__', return_value=None):
            adapter = Qwen3NextModelAdapter(
                model_type=self.model_type,
                model_path=self.model_path
            )

            mock_batch_dataset = [['batch1'], ['batch2']]
            adapter._get_batch_tokenized_data = MagicMock(return_value=mock_batch_dataset)

            result = adapter.handle_dataset_by_batch(
                dataset='test_data',
                batch_size=2,
                device=DeviceType.CPU
            )

            self.assertEqual(result, mock_batch_dataset)
            adapter._get_batch_tokenized_data.assert_called_once_with(
                calib_list='test_data',
                batch_size=2,
                device=DeviceType.CPU
            )

    @patch.dict('sys.modules', {
        'transformers.models.qwen3_next.modeling_qwen3_next': MagicMock(),
    })
    def test_init_model(self):
        """测试init_model方法"""
        from msmodelslim.model.qwen3_next.model_adapter import Qwen3NextModelAdapter
        with patch('msmodelslim.model.qwen3_next.model_adapter.TransformersModel.__init__', return_value=None):
            adapter = Qwen3NextModelAdapter(
                model_type=self.model_type,
                model_path=self.model_path
            )

            mock_model = nn.Linear(10, 10)
            adapter._load_model = MagicMock(return_value=mock_model)

            result = adapter.init_model(device=DeviceType.NPU)

            self.assertIs(result, mock_model)
            adapter._load_model.assert_called_once_with(DeviceType.NPU)

    @patch.dict('sys.modules', {
        'transformers.models.qwen3_next.modeling_qwen3_next': MagicMock(),
    })
    def test_generate_model_visit(self):
        """测试generate_model_visit方法"""
        from msmodelslim.model.qwen3_next.model_adapter import Qwen3NextModelAdapter
        with patch('msmodelslim.model.qwen3_next.model_adapter.TransformersModel.__init__', return_value=None):
            adapter = Qwen3NextModelAdapter(
                model_type=self.model_type,
                model_path=self.model_path
            )

            mock_model = nn.Linear(10, 10)
            mock_transformer_blocks = [('block1', nn.Linear(5, 5)), ('block2', nn.Linear(5, 5))]

            with patch(
                    'msmodelslim.model.qwen3_next.'
                    'model_adapter.generated_decoder_layer_visit_func') as mock_visit_func:
                mock_visit_func.return_value = iter([MagicMock(spec=ProcessRequest)])

                result = list(adapter.generate_model_visit( \
                    model=mock_model, transformer_blocks=mock_transformer_blocks))

                mock_visit_func.assert_called_once_with(mock_model, mock_transformer_blocks)
                self.assertIsInstance(result, list)
                self.assertGreater(len(result), 0)
                self.assertIsInstance(result[0], ProcessRequest)

    @patch.dict('sys.modules', {
        'transformers.models.qwen3_next.modeling_qwen3_next': MagicMock(),
    })
    def test_generate_model_visit_with_none_transformer_blocks(self):
        """测试generate_model_visit方法当transformer_blocks为None时"""
        from msmodelslim.model.qwen3_next.model_adapter import Qwen3NextModelAdapter
        with patch('msmodelslim.model.qwen3_next.model_adapter.TransformersModel.__init__', return_value=None):
            adapter = Qwen3NextModelAdapter(
                model_type=self.model_type,
                model_path=self.model_path
            )

            mock_model = nn.Linear(10, 10)

            with patch(
                    'msmodelslim.model.qwen3_next.model_adapter.'
                    'generated_decoder_layer_visit_func') as mock_visit_func:
                mock_visit_func.return_value = iter([MagicMock(spec=ProcessRequest)])

                result = list(adapter.generate_model_visit(model=mock_model))

                mock_visit_func.assert_called_once_with(mock_model, None)
                self.assertIsInstance(result, list)
                self.assertGreater(len(result), 0)

    @patch.dict('sys.modules', {
        'transformers.models.qwen3_next.modeling_qwen3_next': MagicMock(),
    })
    def test_generate_model_forward(self):
        """测试generate_model_forward方法"""
        from msmodelslim.model.qwen3_next.model_adapter import Qwen3NextModelAdapter
        with patch('msmodelslim.model.qwen3_next.model_adapter.TransformersModel.__init__', return_value=None):
            adapter = Qwen3NextModelAdapter(
                model_type=self.model_type,
                model_path=self.model_path
            )

            mock_model = nn.Linear(10, 10)
            mock_inputs = {'input_ids': [1, 2, 3]}

            with patch(
                    'msmodelslim.model.qwen3_next.model_adapter.'
                    'transformers_generated_forward_func') as mock_forward_func:
                mock_forward_func.return_value = iter([MagicMock(spec=ProcessRequest)])

                result = list(adapter.generate_model_forward(model=mock_model, inputs=mock_inputs))

                mock_forward_func.assert_called_once_with(mock_model, mock_inputs)
                self.assertIsInstance(result, list)
                self.assertGreater(len(result), 0)
                self.assertIsInstance(result[0], ProcessRequest)

    @patch.dict('sys.modules', {
        'transformers.models.qwen3_next.modeling_qwen3_next': MagicMock(),
    })
    def test_enable_kv_cache(self):
        """测试enable_kv_cache方法"""
        from msmodelslim.model.qwen3_next.model_adapter import Qwen3NextModelAdapter
        with patch('msmodelslim.model.qwen3_next.model_adapter.TransformersModel.__init__', return_value=None):
            adapter = Qwen3NextModelAdapter(
                model_type=self.model_type,
                model_path=self.model_path
            )

            mock_model = nn.Linear(10, 10)
            adapter._enable_kv_cache = MagicMock(return_value=None)

            result = adapter.enable_kv_cache(model=mock_model, need_kv_cache=True)

            adapter._enable_kv_cache.assert_called_once_with(mock_model, True)
            self.assertIsNone(result)

    @patch.dict('sys.modules', {
        'transformers.models.qwen3_next.modeling_qwen3_next': MagicMock(),
    })
    def test_get_adapter_config_for_subgraph(self):
        """测试get_adapter_config_for_subgraph方法"""
        from msmodelslim.model.qwen3_next.model_adapter import Qwen3NextModelAdapter
        with patch('msmodelslim.model.qwen3_next.model_adapter.TransformersModel.__init__', return_value=None):
            adapter = Qwen3NextModelAdapter(
                model_type=self.model_type,
                model_path=self.model_path
            )
            # 模拟配置参数
            adapter.config = MagicMock()
            adapter.config.full_attention_interval = 2
            adapter.config.num_hidden_layers = 6

            result = adapter.get_adapter_config_for_subgraph()

            # 验证返回值类型
            self.assertIsInstance(result, list)
            # 验证创建了正确的AdapterConfig和MappingConfig
            self.assertEqual(len(result), 3)  # 1, 3, 5层会匹配

            for i, config in enumerate(result):
                self.assertIsInstance(config, AdapterConfig)
                self.assertEqual(config.subgraph_type, "norm-linear")
                self.assertIsInstance(config.mapping, MappingConfig)
                expected_layer_idx = 2 * i + 1  # 1, 3, 5
                expected_source = f"model.layers.{expected_layer_idx}.input_layernorm"
                self.assertEqual(config.mapping.source, expected_source)

    @patch.dict('sys.modules', {
        'transformers.models.qwen3_next.modeling_qwen3_next': MagicMock(),
    })
    def test_ascendv1_save_module_preprocess_with_input_layernorm(self):
        """测试ascendv1_save_module_preprocess方法当prefix包含input_layernorm且module是Qwen3RMSNorm时"""
        from msmodelslim.model.qwen3_next.model_adapter import Qwen3NextModelAdapter
        with patch('msmodelslim.model.qwen3_next.model_adapter.TransformersModel.__init__', return_value=None):
            adapter = Qwen3NextModelAdapter(
                model_type=self.model_type,
                model_path=self.model_path
            )

            # 创建模拟的Qwen3RMSNorm模块
            mock_module = MagicMock()
            mock_module.__class__.__name__ = 'Qwen3RMSNorm'
            mock_module.weight = MagicMock()
            mock_module.weight.shape = [100]
            mock_module.variance_epsilon = 1e-6
            mock_module.weight.data = MagicMock()
            mock_module.weight.data.__sub__ = MagicMock(return_value=MagicMock())
            mock_module.weight.data.__radd__ = MagicMock(return_value=MagicMock())

            mock_model = MagicMock()

            result = adapter.ascendv1_save_module_preprocess("model.layers.0.input_layernorm", mock_module, mock_model)

            # 验证返回值不为None
            self.assertIsNotNone(result)
            # 验证model.set_submodule被调用
            mock_model.set_submodule.assert_called_once()

    @patch.dict('sys.modules', {
        'transformers.models.qwen3_next.modeling_qwen3_next': MagicMock(),
    })
    def test_ascendv1_save_module_preprocess_without_input_layernorm(self):
        """测试ascendv1_save_module_preprocess方法当prefix不包含input_layernorm时"""
        from msmodelslim.model.qwen3_next.model_adapter import Qwen3NextModelAdapter
        with patch('msmodelslim.model.qwen3_next.model_adapter.TransformersModel.__init__', return_value=None):
            adapter = Qwen3NextModelAdapter(
                model_type=self.model_type,
                model_path=self.model_path
            )

            mock_module = MagicMock()
            mock_model = MagicMock()

            result = adapter.ascendv1_save_module_preprocess("model.layers.0.some_other_module", mock_module,
                                                             mock_model)

            # 验证返回值为None
            self.assertIsNone(result)
            # 验证model.set_submodule没有被调用
            mock_model.set_submodule.assert_not_called()

    @patch.dict('sys.modules', {
        'transformers.models.qwen3_next.modeling_qwen3_next': MagicMock(),
    })
    def test_ascendv1_save_module_preprocess_with_wrong_module_type(self):
        """测试ascendv1_save_module_preprocess方法当module不是Qwen3RMSNorm时"""
        from msmodelslim.model.qwen3_next.model_adapter import Qwen3NextModelAdapter
        with patch('msmodelslim.model.qwen3_next.model_adapter.TransformersModel.__init__', return_value=None):
            adapter = Qwen3NextModelAdapter(
                model_type=self.model_type,
                model_path=self.model_path
            )

            mock_module = MagicMock()
            mock_module.__class__.__name__ = 'SomeOtherModule'
            mock_model = MagicMock()

            result = adapter.ascendv1_save_module_preprocess("model.layers.0.input_layernorm", mock_module, mock_model)

            # 验证返回值为None
            self.assertIsNone(result)
            # 验证model.set_submodule没有被调用
            mock_model.set_submodule.assert_not_called()