# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import torch.nn as nn

from msmodelslim.core.const import DeviceType
from msmodelslim.model.default.model_adapter import DefaultModelAdapter
from msmodelslim.utils.exception import InvalidModelError


class TestDefaultModelAdapter(unittest.TestCase):

    def setUp(self):
        self.model_type = 'DefaultModel'
        self.model_path = Path('.')
        self.trust_remote_code = False

    @patch('msmodelslim.model.default.model_adapter.TransformersModel.__init__')
    def test_initialization_success(self, mock_super_init):
        """测试默认模型适配器成功初始化"""
        mock_super_init.return_value = None

        with patch('msmodelslim.model.default.model_adapter.get_logger') as mock_logger:
            adapter = DefaultModelAdapter(
                model_type=self.model_type,
                model_path=self.model_path,
                trust_remote_code=self.trust_remote_code
            )

            mock_logger().warning.assert_called_once()
            self.assertIn('default model adapter',
                          mock_logger().warning.call_args[0][0])
            mock_super_init.assert_called_once_with(
                self.model_type,
                self.model_path,
                self.trust_remote_code
            )

    @patch('msmodelslim.model.default.model_adapter.TransformersModel.__init__')
    def test_initialization_failure(self, mock_super_init):
        """测试默认模型适配器初始化失败"""
        mock_super_init.side_effect = Exception("Model loading failed")

        with self.assertRaises(InvalidModelError):
            DefaultModelAdapter(
                model_type=self.model_type,
                model_path=self.model_path,
                trust_remote_code=self.trust_remote_code
            )

    def test_get_model_type(self):
        """测试get_model_type方法"""
        with patch('msmodelslim.model.default.model_adapter.TransformersModel.__init__', return_value=None):
            adapter = DefaultModelAdapter(
                model_type=self.model_type,
                model_path=self.model_path
            )
            adapter.model_type = self.model_type

            with patch('msmodelslim.model.default.model_adapter.get_logger') as mock_logger:
                result = adapter.get_model_type()

                self.assertEqual(result, self.model_type)
                mock_logger().warning.assert_called_once()
                self.assertIn('default get_model_type',
                              mock_logger().warning.call_args[0][0])

    def test_get_model_pedigree(self):
        """测试get_model_pedigree方法"""
        with patch('msmodelslim.model.default.model_adapter.TransformersModel.__init__', return_value=None):
            adapter = DefaultModelAdapter(
                model_type=self.model_type,
                model_path=self.model_path
            )
            adapter.model_pedigree = 'default_pedigree'

            with patch('msmodelslim.model.default.model_adapter.get_logger') as mock_logger:
                result = adapter.get_model_pedigree()

                self.assertEqual(result, 'default_pedigree')
                mock_logger().warning.assert_called_once()
                self.assertIn('default get_model_pedigree',
                              mock_logger().warning.call_args[0][0])

    def test_load_model_success(self):
        """测试load_model方法成功情况"""
        with patch('msmodelslim.model.default.model_adapter.TransformersModel.__init__', return_value=None):
            adapter = DefaultModelAdapter(
                model_type=self.model_type,
                model_path=self.model_path
            )

            mock_model = nn.Linear(10, 10)
            adapter._load_model = MagicMock(return_value=mock_model)

            with patch('msmodelslim.model.default.model_adapter.get_logger') as mock_logger:
                result = adapter.load_model(device=DeviceType.CPU)

                self.assertIs(result, mock_model)
                adapter._load_model.assert_called_once_with(DeviceType.CPU)
                mock_logger().warning.assert_called()

    def test_load_model_failure(self):
        """测试load_model方法失败情况"""
        with patch('msmodelslim.model.default.model_adapter.TransformersModel.__init__', return_value=None):
            adapter = DefaultModelAdapter(
                model_type=self.model_type,
                model_path=self.model_path
            )

            adapter._load_model = MagicMock(side_effect=Exception("Loading failed"))

            with self.assertRaises(InvalidModelError):
                adapter.load_model(device=DeviceType.CPU)

    def test_handle_dataset_success(self):
        """测试handle_dataset方法成功情况"""
        with patch('msmodelslim.model.default.model_adapter.TransformersModel.__init__', return_value=None):
            adapter = DefaultModelAdapter(
                model_type=self.model_type,
                model_path=self.model_path
            )

            mock_dataset = ['data1', 'data2']
            adapter._get_tokenized_data = MagicMock(return_value=mock_dataset)

            with patch('msmodelslim.model.default.model_adapter.get_logger') as mock_logger:
                result = adapter.handle_dataset(dataset='test_data', device=DeviceType.CPU)

                self.assertEqual(result, mock_dataset)
                adapter._get_tokenized_data.assert_called_once_with('test_data', DeviceType.CPU)
                mock_logger().warning.assert_called()

    def test_handle_dataset_failure(self):
        """测试handle_dataset方法失败情况"""
        with patch('msmodelslim.model.default.model_adapter.TransformersModel.__init__', return_value=None):
            adapter = DefaultModelAdapter(
                model_type=self.model_type,
                model_path=self.model_path
            )

            adapter._get_tokenized_data = MagicMock(side_effect=Exception("Processing failed"))

            with self.assertRaises(InvalidModelError):
                adapter.handle_dataset(dataset='test_data', device=DeviceType.CPU)

    def test_handle_dataset_by_batch_success(self):
        """测试handle_dataset_by_batch方法成功情况"""
        with patch('msmodelslim.model.default.model_adapter.TransformersModel.__init__', return_value=None):
            adapter = DefaultModelAdapter(
                model_type=self.model_type,
                model_path=self.model_path
            )

            mock_batch_dataset = [['batch1_data1', 'batch1_data2'], ['batch2_data1', 'batch2_data2']]
            adapter._get_batch_tokenized_data = MagicMock(return_value=mock_batch_dataset)

            with patch('msmodelslim.model.default.model_adapter.get_logger') as mock_logger:
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
                mock_logger().warning.assert_called()

    def test_init_model_success(self):
        """测试init_model方法成功情况"""
        with patch('msmodelslim.model.default.model_adapter.TransformersModel.__init__', return_value=None):
            adapter = DefaultModelAdapter(
                model_type=self.model_type,
                model_path=self.model_path
            )

            mock_model = nn.Linear(10, 10)
            adapter._load_model = MagicMock(return_value=mock_model)

            with patch('msmodelslim.model.default.model_adapter.get_logger') as mock_logger:
                result = adapter.init_model(device=DeviceType.NPU)

                self.assertIs(result, mock_model)
                adapter._load_model.assert_called_once_with(DeviceType.NPU)

    def test_enable_kv_cache_success(self):
        """测试enable_kv_cache方法成功情况"""
        with patch('msmodelslim.model.default.model_adapter.TransformersModel.__init__', return_value=None):
            adapter = DefaultModelAdapter(
                model_type=self.model_type,
                model_path=self.model_path
            )

            mock_model = nn.Linear(10, 10)
            adapter._enable_kv_cache = MagicMock(return_value=None)

            with patch('msmodelslim.model.default.model_adapter.get_logger') as mock_logger:
                result = adapter.enable_kv_cache(model=mock_model, need_kv_cache=True)

                adapter._enable_kv_cache.assert_called_once_with(mock_model, True)
                mock_logger().warning.assert_called()

    def test_enable_kv_cache_failure(self):
        """测试enable_kv_cache方法失败情况"""
        with patch('msmodelslim.model.default.model_adapter.TransformersModel.__init__', return_value=None):
            adapter = DefaultModelAdapter(
                model_type=self.model_type,
                model_path=self.model_path
            )

            mock_model = nn.Linear(10, 10)
            adapter._enable_kv_cache = MagicMock(side_effect=Exception("Enable KV cache failed"))

            with self.assertRaises(InvalidModelError):
                adapter.enable_kv_cache(model=mock_model, need_kv_cache=True)
