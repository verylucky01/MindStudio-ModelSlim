# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import torch.nn as nn

from msmodelslim.core.const import DeviceType
from msmodelslim.model.qwq.model_adapter import QwqModelAdapter


class TestQwqModelAdapter(unittest.TestCase):
    """测试QwqModelAdapter的功能"""

    def setUp(self):
        """测试前的准备工作"""
        self.model_type = 'QwQ-32B'
        self.model_path = Path('.')
        self.trust_remote_code = False

    def test_get_model_type_when_initialized_then_return_model_type(self):
        """测试get_model_type方法：初始化后应返回正确的模型类型"""
        with patch('msmodelslim.model.qwq.model_adapter.TransformersModel.__init__', return_value=None):
            adapter = QwqModelAdapter(
                model_type=self.model_type,
                model_path=self.model_path,
                trust_remote_code=self.trust_remote_code
            )
            adapter.model_type = self.model_type

            result = adapter.get_model_type()

            self.assertEqual(result, self.model_type)

    def test_get_model_pedigree_when_called_then_return_qwq(self):
        """测试get_model_pedigree方法：应返回'qwq'"""
        with patch('msmodelslim.model.qwq.model_adapter.TransformersModel.__init__', return_value=None):
            adapter = QwqModelAdapter(
                model_type=self.model_type,
                model_path=self.model_path
            )

            result = adapter.get_model_pedigree()

            self.assertEqual(result, 'qwq')

    def test_load_model_with_npu_device_when_called_then_delegate_to_load_model(self):
        """测试load_model方法：使用NPU设备时应委托给_load_model方法"""
        with patch('msmodelslim.model.qwq.model_adapter.TransformersModel.__init__', return_value=None):
            adapter = QwqModelAdapter(
                model_type=self.model_type,
                model_path=self.model_path
            )

            mock_model = nn.Linear(10, 10)
            adapter._load_model = MagicMock(return_value=mock_model)

            result = adapter.load_model(device=DeviceType.NPU)

            self.assertIs(result, mock_model)
            adapter._load_model.assert_called_once_with(DeviceType.NPU)

    def test_handle_dataset_when_called_then_return_tokenized_data(self):
        """测试handle_dataset方法：应返回tokenized数据"""
        with patch('msmodelslim.model.qwq.model_adapter.TransformersModel.__init__', return_value=None):
            adapter = QwqModelAdapter(
                model_type=self.model_type,
                model_path=self.model_path
            )

            mock_dataset = ['data1', 'data2']
            adapter._get_tokenized_data = MagicMock(return_value=mock_dataset)

            result = adapter.handle_dataset(dataset='test_data', device=DeviceType.CPU)

            self.assertEqual(result, mock_dataset)
            adapter._get_tokenized_data.assert_called_once_with('test_data', DeviceType.CPU)

    def test_handle_dataset_by_batch_when_called_then_return_batch_tokenized_data(self):
        """测试handle_dataset_by_batch方法：应返回批量tokenized数据"""
        with patch('msmodelslim.model.qwq.model_adapter.TransformersModel.__init__', return_value=None):
            adapter = QwqModelAdapter(
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

    def test_init_model_when_called_then_delegate_to_load_model(self):
        """测试init_model方法：应委托给_load_model方法"""
        with patch('msmodelslim.model.qwq.model_adapter.TransformersModel.__init__', return_value=None):
            adapter = QwqModelAdapter(
                model_type=self.model_type,
                model_path=self.model_path
            )

            mock_model = nn.Linear(10, 10)
            adapter._load_model = MagicMock(return_value=mock_model)

            result = adapter.init_model(device=DeviceType.NPU)

            self.assertIs(result, mock_model)
            adapter._load_model.assert_called_once_with(DeviceType.NPU)

    def test_enable_kv_cache_when_called_with_true_then_enable_cache(self):
        """测试enable_kv_cache方法：传入True时应启用缓存"""
        with patch('msmodelslim.model.qwq.model_adapter.TransformersModel.__init__', return_value=None):
            adapter = QwqModelAdapter(
                model_type=self.model_type,
                model_path=self.model_path
            )

            mock_model = nn.Linear(10, 10)
            adapter._enable_kv_cache = MagicMock(return_value=None)

            result = adapter.enable_kv_cache(model=mock_model, need_kv_cache=True)

            # 验证_enable_kv_cache被调用
            adapter._enable_kv_cache.assert_called_once_with(mock_model, True)

    def test_enable_kv_cache_when_called_with_false_then_disable_cache(self):
        """测试enable_kv_cache方法：传入False时应禁用缓存"""
        with patch('msmodelslim.model.qwq.model_adapter.TransformersModel.__init__', return_value=None):
            adapter = QwqModelAdapter(
                model_type=self.model_type,
                model_path=self.model_path
            )

            mock_model = nn.Linear(10, 10)
            adapter._enable_kv_cache = MagicMock(return_value=None)

            adapter.enable_kv_cache(model=mock_model, need_kv_cache=False)

            # 验证参数正确传递
            adapter._enable_kv_cache.assert_called_once_with(mock_model, False)

    def test_handle_dataset_with_empty_dataset_when_called_then_return_empty_result(self):
        """测试handle_dataset方法：空数据集时应返回空结果"""
        with patch('msmodelslim.model.qwq.model_adapter.TransformersModel.__init__', return_value=None):
            adapter = QwqModelAdapter(
                model_type=self.model_type,
                model_path=self.model_path
            )

            mock_dataset = []
            adapter._get_tokenized_data = MagicMock(return_value=mock_dataset)

            result = adapter.handle_dataset(dataset='', device=DeviceType.CPU)

            self.assertEqual(result, [])
