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

import functools
import os
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock, call

import pytest

from msmodelslim.app.quant_service.interface import BaseQuantConfig
from msmodelslim.app.quant_service.multimodal_sd_v1.quant_service import (
    MultimodalSDModelslimV1QuantService,
    MultimodalSDModelslimV1QuantConfig,
    MultimodalPipelineInterface,
    DeviceType,
    SchemaValidateError
)


class TestQuantProcessComplete:
    mock_model = None
    mock_model1 = None
    mock_model2 = None

    def setup_method(self):
        # 基础服务与核心依赖
        self.dataset_loader = Mock()
        self.service = MultimodalSDModelslimV1QuantService(self.dataset_loader)

        # 1. 简化配置层级：用MagicMock自动支持属性访问
        self.mock_quant_spec = MagicMock()
        # 量化配置核心属性
        self.mock_quant_spec.multimodal_sd_config.model_extra = {"model_config": "test_config"}
        self.mock_quant_spec.multimodal_sd_config.dump_config = MagicMock(
            capture_mode='args', dump_data_dir="/test/dump"
        )
        # 处理/保存配置（2个实例，覆盖多配置场景）
        self.mock_quant_spec.process = [Mock(), Mock()]
        self.mock_quant_spec.save = [Mock(), Mock()]
        self.mock_quant_spec.runner = "layer_wise"

        # 2. 只保留BaseQuantConfig的spec（推荐，符合实际类型约束）
        self.quant_config = Mock(spec=BaseQuantConfig)
        # 给quant_config添加spec属性，模拟原意图（关联mock_quant_spec）
        self.quant_config.spec = self.mock_quant_spec

        # 模型适配器（保持原逻辑）
        self.model_adapter = Mock(spec=MultimodalPipelineInterface)
        self.mock_model = Mock()
        self.mock_model1 = Mock()
        self.mock_model2 = Mock()
        self.model_adapter.init_model.return_value = {'': self.mock_model}
        self.model_adapter.model_args = Mock()
        self.model_adapter.model_args.task_config = ''
        self.model_adapter.transformer = Mock()

        # 3. 测试固定参数
        self.save_path = Path("/test/save")
        self.device = DeviceType.NPU

    def test_backend_name(self):
        """验证后端名称"""
        assert self.service.backend_name == "multimodal_sd_modelslim_v1"

    @pytest.mark.parametrize("case", [
        # 参数化合并4个无效参数测试
        ("invalid_quant_config", Mock(), Mock(spec=MultimodalPipelineInterface), Path("test"),
         "task must be a BaseTask"),
        ("invalid_model_adapter", Mock(spec=BaseQuantConfig), Mock(), Path("test"),
         "model must be a MultimodalPipelineInterface"),
        ("invalid_save_path", Mock(spec=BaseQuantConfig), Mock(spec=MultimodalPipelineInterface), "invalid_path",
         "save_path must be a Path or None"),
        ("invalid_device", Mock(spec=BaseQuantConfig), Mock(spec=MultimodalPipelineInterface), Path("test"),
         "device must be a DeviceType"),
    ])
    def test_quantize_invalid_params(self, case):
        """参数化测试：合并所有无效参数场景"""
        name, quant_cfg, model_adap, save_p, err_msg = case
        with pytest.raises(SchemaValidateError) as excinfo:
            # 只有invalid_device场景传入无效device，其他用合法DeviceType
            device = "invalid" if name == "invalid_device" else DeviceType.NPU
            self.service.quantize(quant_cfg, model_adap, save_p, device=device)
        assert err_msg in str(excinfo.value)

    @patch.object(MultimodalSDModelslimV1QuantConfig, 'from_base')
    def test_quantize_normal_flow(self, mock_from_base):
        """测试正常量化流程"""
        mock_from_base.return_value = self.quant_config
        self.service.quant_process = Mock(return_value="result")

        # 执行测试
        self.service.quantize(
            Mock(spec=BaseQuantConfig), self.model_adapter, self.save_path, self.device
        )

        # 核心验证：配置转换+流程调用
        mock_from_base.assert_called_once()
        self.service.quant_process.assert_called_once_with(
            self.quant_config, self.model_adapter, self.save_path, self.device
        )

    @patch("msmodelslim.app.quant_service.multimodal_sd_v1.quant_service.load_cached_data_for_models")
    @patch("msmodelslim.app.quant_service.multimodal_sd_v1.quant_service.to_device")
    @patch("msmodelslim.app.quant_service.multimodal_sd_v1.quant_service.LayerWiseRunner")
    def test_quant_process_full_flow(self, mock_runner_cls, mock_to_device, mock_load_cache):
        """简化完整流程测试：聚焦核心步骤"""
        # 1. 简化依赖Mock行为
        self.model_adapter.init_model.return_value = {'model1': self.mock_model1, 'model2': self.mock_model2}
        self.model_adapter.model_args.task_config = 'mock'
        mock_load_cache.return_value = "raw_calib_data"
        calib_data = {'model1': '/test/dump/calib_data_mock_model1.pth',
                      'model2': '/test/dump/calib_data_mock_model2.pth'}
        mock_to_device.return_value = calib_data
        mock_runner = Mock()
        mock_runner_cls.return_value = mock_runner

        # 2. 执行测试
        self.service.quant_process(self.quant_config, self.model_adapter, self.save_path, self.device)

        # 3. 核心步骤验证（按执行顺序）
        # 模型配置与加载
        self.model_adapter.set_model_args.assert_called_once_with("test_config")
        self.model_adapter.load_pipeline.assert_called_once()
        # 数据加载与设备转换
        pth_file_path_list = {'model1': '/test/dump/calib_data_mock_model1.pth',
                              'model2': '/test/dump/calib_data_mock_model2.pth'}
        model = {'model1': self.mock_model1, 'model2': self.mock_model2}
        mock_load_cache.assert_called_once_with(
            pth_file_path_list=pth_file_path_list,
            generate_func=self.model_adapter.run_calib_inference,
            models=model,
            dump_config=self.mock_quant_spec.multimodal_sd_config.dump_config
        )
        mock_to_device.assert_called_once_with("raw_calib_data", self.device.value)
        # 保存配置与Runner处理
        expert_save_path1 = self.save_path.joinpath(f"{self.model_adapter.model_args.task_config}_model1")
        expert_save_path2 = self.save_path.joinpath(f"{self.model_adapter.model_args.task_config}_model2")
        for save_cfg in self.mock_quant_spec.save:
            save_cfg.set_save_directory.assert_has_calls(
                [call(expert_save_path1), call(expert_save_path2)],
                any_order=False
            )
        mock_runner.add_processor.assert_has_calls(
            [call(processor_cfg=cfg) for cfg in self.mock_quant_spec.process],
            any_order=False
        )
        # 量化应用（验证partial函数参数）
        partial_func = self.model_adapter.apply_quantization.call_args[0][0]
        assert isinstance(partial_func, functools.partial)
        assert partial_func.func == mock_runner.run
        assert partial_func.keywords == {"calib_data": "/test/dump/calib_data_mock_model2.pth",
                                         "device": self.device, "model": self.mock_model2}

    @patch("msmodelslim.app.quant_service.multimodal_sd_v1.quant_service.load_cached_data_for_models")
    @patch("msmodelslim.app.quant_service.multimodal_sd_v1.quant_service.to_device")
    @patch("msmodelslim.app.quant_service.multimodal_sd_v1.quant_service.LayerWiseRunner")
    def test_quant_process_full_flow_not_moe(self, mock_runner_cls, mock_to_device, mock_load_cache):
        """简化完整流程测试：聚焦核心步骤"""
        # 1. 简化依赖Mock行为
        self.model_adapter.init_model.return_value = {'': self.mock_model1}
        self.model_adapter.model_args.task_config = ''
        mock_load_cache.return_value = "raw_calib_data"
        calib_data = {'': '/test/dump/calib_data__.pth'}
        mock_to_device.return_value = calib_data
        mock_runner = Mock()
        mock_runner_cls.return_value = mock_runner

        # 2. 执行测试
        self.service.quant_process(self.quant_config, self.model_adapter, self.save_path, self.device)

        # 3. 核心步骤验证（按执行顺序）
        # 模型配置与加载
        self.model_adapter.set_model_args.assert_called_once_with("test_config")
        self.model_adapter.load_pipeline.assert_called_once()
        # 数据加载与设备转换
        pth_file_path_list = {'': '/test/dump/calib_data__.pth'}
        model = {'': self.mock_model1}
        mock_load_cache.assert_called_once_with(
            pth_file_path_list=pth_file_path_list,
            generate_func=self.model_adapter.run_calib_inference,
            models=model,
            dump_config=self.mock_quant_spec.multimodal_sd_config.dump_config
        )
        mock_to_device.assert_called_once_with("raw_calib_data", self.device.value)
        # 保存配置与Runner处理
        expert_save_path1 = self.save_path
        for save_cfg in self.mock_quant_spec.save:
            save_cfg.set_save_directory.assert_has_calls(
                [call(expert_save_path1)],
                any_order=False
            )
        mock_runner.add_processor.assert_has_calls(
            [call(processor_cfg=cfg) for cfg in self.mock_quant_spec.process],
            any_order=False
        )
        # 量化应用（验证partial函数参数）
        partial_func = self.model_adapter.apply_quantization.call_args[0][0]
        assert isinstance(partial_func, functools.partial)
        assert partial_func.func == mock_runner.run
        assert partial_func.keywords == {"calib_data": "/test/dump/calib_data__.pth",
                                         "device": self.device, "model": self.mock_model1}

    @patch("msmodelslim.app.quant_service.multimodal_sd_v1.quant_service.load_cached_data_for_models")
    @patch("msmodelslim.app.quant_service.multimodal_sd_v1.quant_service.get_logger")
    @patch("msmodelslim.app.quant_service.multimodal_sd_v1.quant_service.LayerWiseRunner")
    def test_quant_process_special_scenarios(self, mock_runner_cls, mock_get_logger, mock_load_cache):
        """合并特殊场景测试：彻底隔离Mock调用记录，避免干扰"""
        # 1. 初始化依赖
        mock_load_cache.return_value = {"": "raw_calib_data"}
        mock_logger = mock_get_logger.return_value
        mock_runner_cls.return_value = Mock()
        dump_config = self.mock_quant_spec.multimodal_sd_config.dump_config

        # -------------------------- 场景1：默认路径（dump_data_dir为空，save_path有效）--------------------------
        # 初始化场景1的save_config（独立实例，避免干扰）
        scene1_save_cfgs = [Mock(), Mock()]
        self.mock_quant_spec.save = scene1_save_cfgs  # 替换为场景1专属的save_config

        dump_config.dump_data_dir = ""
        self.service.quant_process(self.quant_config, self.model_adapter, self.save_path, self.device)

        # 验证场景1的save_config被调用（符合预期）
        for save_cfg in scene1_save_cfgs:
            save_cfg.set_save_directory.assert_called_once_with(self.save_path)

        # -------------------------- 场景2：非layer_wise runner --------------------------
        # 初始化场景2的save_config（新实例，隔离场景1）
        scene2_save_cfgs = [Mock(), Mock()]
        self.mock_quant_spec.save = scene2_save_cfgs

        self.mock_quant_spec.runner = "other_runner"
        self.service.quant_process(self.quant_config, self.model_adapter, self.save_path, self.device)

        # 验证场景2的save_config被调用（save_path有效，符合预期）
        for save_cfg in scene2_save_cfgs:
            save_cfg.set_save_directory.assert_called_once_with(self.save_path)
        # 验证警告日志
        mock_logger.warning.assert_has_calls(
            [call("runner for multimodal_sd_v1 is not layer_wise, will be converted to layer_wise.")]
        )

        # -------------------------- 场景3：无save_path --------------------------
        # 重置关键Mock：避免前场景干扰
        mock_load_cache.reset_mock()
        scene3_save_cfgs = [Mock(), Mock()]
        self.mock_quant_spec.save = scene3_save_cfgs
        dump_config.dump_data_dir = "./test"

        # 执行无save_path的流程
        self.service.quant_process(self.quant_config, self.model_adapter, save_path=None, device=self.device)
        for save_cfg in scene3_save_cfgs:
            save_cfg.set_save_directory.assert_not_called()
