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

import os
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock

from testing_utils.mock import mock_init_config

from msmodelslim.core.const import DeviceType, QuantType
from msmodelslim.core.practice.interface import Metadata, PracticeConfig
from msmodelslim.utils.exception import SchemaValidateError, ToDoError, UnsupportedError

mock_init_config()


class TestNaiveQuantizationAppBase(unittest.TestCase):
    """NaiveQuantizationApplication 测试基类"""

    def setUp(self):
        original_umask = os.umask(0)
        try:
            os.umask(0o026)
            self.temp_dir = tempfile.mkdtemp()
            self.model_path = Path(self.temp_dir) / "model"
            self.model_path.mkdir()
            self.save_path = Path(self.temp_dir) / "save"
            self.save_path.mkdir()
        finally:
            os.umask(original_umask)

    def tearDown(self):
        if hasattr(self, 'temp_dir') and os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)


class TestNaiveQuantizationApplicationInit(TestNaiveQuantizationAppBase):
    """测试 NaiveQuantizationApplication 初始化"""

    def test_init(self):
        """测试初始化"""
        from msmodelslim.app.naive_quantization.application import NaiveQuantizationApplication

        mock_practice_manager = MagicMock()
        mock_quant_service = MagicMock()
        mock_model_factory = MagicMock()

        app = NaiveQuantizationApplication(
            practice_manager=mock_practice_manager,
            quant_service=mock_quant_service,
            model_factory=mock_model_factory,
        )

        self.assertEqual(app.practice_manager, mock_practice_manager)
        self.assertEqual(app.quant_service, mock_quant_service)
        self.assertEqual(app.model_factory, mock_model_factory)


class TestCheckConfig(TestNaiveQuantizationAppBase):
    """测试 check_config 静态方法"""

    def _make_config(self, w_bit=8, a_bit=8, is_sparse=False, kv_cache=False,
                     verified_model_types=None, verified_tags=None):
        metadata = Metadata(
            config_id="test_config",
            score=90,
            label={"w_bit": w_bit, "a_bit": a_bit, "is_sparse": is_sparse, "kv_cache": kv_cache},
            verified_model_types=verified_model_types or [],
            verified_tags=verified_tags or {},
        )
        return PracticeConfig(apiversion="modelslim_v1", metadata=metadata)

    def test_check_config_label_mismatch_w_bit(self):
        """测试 labels 不匹配 w_bit"""
        from msmodelslim.app.naive_quantization.application import NaiveQuantizationApplication

        config = self._make_config(w_bit=4)
        result = NaiveQuantizationApplication.check_config(
            config, "Qwen2.5-7B", QuantType.W8A8
        )
        self.assertFalse(result)

    def test_check_config_label_mismatch_a_bit(self):
        """测试 labels 不匹配 a_bit"""
        from msmodelslim.app.naive_quantization.application import NaiveQuantizationApplication

        config = self._make_config(a_bit=4)
        result = NaiveQuantizationApplication.check_config(
            config, "Qwen2.5-7B", QuantType.W8A8
        )
        self.assertFalse(result)

    def test_check_config_verified_model_types_match(self):
        """测试 verified_model_types 匹配时直接返回 True"""
        from msmodelslim.app.naive_quantization.application import NaiveQuantizationApplication

        config = self._make_config(verified_model_types=["Qwen2.5-7B"])
        result = NaiveQuantizationApplication.check_config(
            config, "Qwen2.5-7B", QuantType.W8A8
        )
        self.assertTrue(result)

    def test_check_config_verified_model_types_no_match(self):
        """测试 verified_model_types 不匹配时返回 False"""
        from msmodelslim.app.naive_quantization.application import NaiveQuantizationApplication

        config = self._make_config(verified_model_types=["Other-Model"])
        result = NaiveQuantizationApplication.check_config(
            config, "Qwen2.5-7B", QuantType.W8A8
        )
        self.assertFalse(result)

    def test_check_config_scenario_tags_match(self):
        """测试 scenario_tags 匹配 verified_tags"""
        from msmodelslim.app.naive_quantization.application import NaiveQuantizationApplication

        config = self._make_config(
            verified_tags={"Qwen2.5-7B": [["mindie", "npu"], ["vllm", "cpu"]]}
        )
        result = NaiveQuantizationApplication.check_config(
            config, "Qwen2.5-7B", QuantType.W8A8,
            scenario_tags=["mindie", "npu"],
        )
        self.assertTrue(result)

    def test_check_config_scenario_tags_no_match_returns_standby(self):
        """测试 scenario_tags 不匹配时返回 standby"""
        from msmodelslim.app.naive_quantization.application import (
            NaiveQuantizationApplication,
            STANDBY_CONFIG,
        )

        config = self._make_config(
            verified_tags={"Qwen2.5-7B": [["mindie", "npu"]]}
        )
        result = NaiveQuantizationApplication.check_config(
            config, "Qwen2.5-7B", QuantType.W8A8,
            scenario_tags=["vllm"],
        )
        self.assertEqual(result, STANDBY_CONFIG)

    def test_check_config_no_scenario_tags_returns_true(self):
        """测试无 scenario_tags 时返回 True"""
        from msmodelslim.app.naive_quantization.application import NaiveQuantizationApplication

        config = self._make_config(verified_tags={"Qwen2.5-7B": [["mindie", "npu"]]})
        result = NaiveQuantizationApplication.check_config(
            config, "Qwen2.5-7B", QuantType.W8A8,
            scenario_tags=None,
        )
        self.assertTrue(result)


class TestGetBestPractice(TestNaiveQuantizationAppBase):
    """测试 get_best_practice"""

    def test_get_best_practice_with_config_path(self):
        """测试指定 config_path 时直接返回配置"""
        from msmodelslim.app.naive_quantization.application import NaiveQuantizationApplication

        config_yaml = """apiversion: modelslim_v1
metadata:
  config_id: test_w8a8
  score: 90
  label:
    w_bit: 8
    a_bit: 8
    is_sparse: false
    kv_cache: false
spec: {}
"""
        config_file = Path(self.temp_dir) / "config.yaml"        
        config_file.write_text(config_yaml, encoding="utf-8")

        mock_practice_manager = MagicMock()
        mock_quant_service = MagicMock()
        mock_model_factory = MagicMock()
        mock_model_adapter = MagicMock()

        app = NaiveQuantizationApplication(
            practice_manager=mock_practice_manager,
            quant_service=mock_quant_service,
            model_factory=mock_model_factory,
        )

        config = app.get_best_practice(
            model_adapter=mock_model_adapter,
            config_path=config_file,
        )

        self.assertEqual(config.metadata.config_id, "test_w8a8")
        self.assertEqual(config.metadata.label["w_bit"], 8)
        mock_practice_manager.iter_config.assert_not_called()

    def test_get_best_practice_requires_model_info_interface(self):
        """测试 get_best_practice 要求 model_adapter 实现 ModelInfoInterface"""
        from msmodelslim.app.naive_quantization.application import NaiveQuantizationApplication

        mock_practice_manager = MagicMock()
        mock_practice_manager.__contains__ = MagicMock(return_value=True)
        mock_quant_service = MagicMock()
        mock_model_factory = MagicMock()
        mock_model_adapter = MagicMock()  # 不实现 ModelInfoInterface

        app = NaiveQuantizationApplication(
            practice_manager=mock_practice_manager,
            quant_service=mock_quant_service,
            model_factory=mock_model_factory,
        )

        with self.assertRaises(ToDoError):
            app.get_best_practice(
                model_adapter=mock_model_adapter,
                quant_type=QuantType.W8A8,
            )


class TestQuantParameterValidation(TestNaiveQuantizationAppBase):
    """测试 quant 方法参数校验"""

    def _make_app(self):
        from msmodelslim.app.naive_quantization.application import NaiveQuantizationApplication

        return NaiveQuantizationApplication(
            practice_manager=MagicMock(),
            quant_service=MagicMock(),
            model_factory=MagicMock(),
        )

    def test_quant_model_type_validation(self):
        """测试 model_type 参数校验"""
        app = self._make_app()
        with self.assertRaises(SchemaValidateError):
            app.quant(
                model_type=123,
                model_path=str(self.model_path),
                save_path=str(self.save_path),
            )

    def test_quant_device_type_validation(self):
        """测试 device_type 参数校验"""
        app = self._make_app()
        with self.assertRaises(SchemaValidateError):
            app.quant(
                model_type="Qwen2.5-7B",
                model_path=str(self.model_path),
                save_path=str(self.save_path),
                device_type="invalid",
            )

    def test_quant_quant_type_and_config_path_mutual_exclusion(self):
        """测试 quant_type 与 config_path 互斥"""
        app = self._make_app()
        config_file = Path(self.temp_dir) / "config.yaml"
        config_file.write_text("apiversion: modelslim_v1\nmetadata:\n  config_id: x\n  label: {}\nspec: {}", encoding="utf-8")

        with self.assertRaises(SchemaValidateError):
            app.quant(
                model_type="Qwen2.5-7B",
                model_path=str(self.model_path),
                save_path=str(self.save_path),
                quant_type=QuantType.W8A8,
                config_path=str(config_file),
            )

    def test_quant_trust_remote_code_validation(self):
        """测试 trust_remote_code 参数校验"""
        app = self._make_app()
        with self.assertRaises(SchemaValidateError):
            app.quant(
                model_type="Qwen2.5-7B",
                model_path=str(self.model_path),
                save_path=str(self.save_path),
                trust_remote_code="not_bool",
            )

    def test_quant_scenario_validation(self):
        """测试 tag 参数校验"""
        app = self._make_app()
        with self.assertRaises(SchemaValidateError):
            app.quant(
                model_type="Qwen2.5-7B",
                model_path=str(self.model_path),
                save_path=str(self.save_path),
                tag="not_a_list",
            )


if __name__ == "__main__":
    unittest.main()
