#! /usr/bin/env python3
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

"""
Unit tests for `multimodal_vlm_v1.quant_service`.

The tests are designed to:
- Cover parameter validation in `quantize`.
- Exercise runner type selection logic.
- Validate the overall control flow of `quant_process` without touching heavy
  external dependencies (models, real datasets, actual NPU environment).
"""

from pathlib import Path
from typing import List
from unittest.mock import Mock, MagicMock, patch, call

import pytest

from msmodelslim.core.quant_service.interface import BaseQuantConfig
from msmodelslim.core.quant_service.multimodal_vlm_v1.quant_service import (
    MultimodalVLMModelslimV1QuantService,
    MultimodalVLMModelslimV1QuantConfig,
)
from msmodelslim.core.const import RunnerType, DeviceType
from msmodelslim.core.runner.pipeline_interface import PipelineInterface
from msmodelslim.utils.exception import SchemaValidateError


class TestMultimodalVLMModelslimV1QuantService:
    """测试 `MultimodalVLMModelslimV1QuantService` 的关键行为。"""

    def setup_method(self):
        # 轻量级 dataset_loader mock，仅暴露测试需要的接口
        self.dataset_loader = Mock()
        self.dataset_loader.get_dataset_by_name.return_value = "mock_dataset"

        self.service = MultimodalVLMModelslimV1QuantService(self.dataset_loader)

        # 模型适配器使用 PipelineInterface 的 spec，确保类型检查分支可被命中
        self.model_adapter = Mock(spec=PipelineInterface)

        # 公共参数
        self.save_path = Path("/tmp/quant")
        self.device = DeviceType.NPU

    def test_backend_name_constant(self):
        """验证 backend_name 常量值是否符合预期。"""
        assert self.service.backend_name == "multimodal_vlm_modelslim_v1"

    @pytest.mark.parametrize(
        "case_name, quant_cfg, model_adap, save_p, device, expected_msg",
        [
            (
                "invalid_quant_config",
                Mock(),  # 非 BaseQuantConfig
                Mock(spec=PipelineInterface),
                Path("test"),
                DeviceType.NPU,
                "task is not a BaseTask",
            ),
            (
                "invalid_model_adapter",
                Mock(spec=BaseQuantConfig),
                Mock(),  # 非 PipelineInterface
                Path("test"),
                DeviceType.NPU,
                "model_adapter must be a PipelineInterface",
            ),
            (
                "invalid_save_path",
                Mock(spec=BaseQuantConfig),
                Mock(spec=PipelineInterface),
                "not-a-path",  # type: ignore[arg-type]
                DeviceType.NPU,
                "save_path must be a Path or None",
            ),
            (
                "invalid_device",
                Mock(spec=BaseQuantConfig),
                Mock(spec=PipelineInterface),
                Path("test"),
                "invalid-device",  # type: ignore[arg-type]
                "device must be a DeviceType",
            ),
        ],
    )
    def test_quantize_invalid_params(self, case_name, quant_cfg, model_adap, save_p, device, expected_msg):
        """参数化测试：验证所有非法参数场景都会抛出 SchemaValidateError。"""
        with pytest.raises(SchemaValidateError) as exc_info:
            self.service.quantize(quant_cfg, model_adap, save_p, device=device)
        assert expected_msg in str(exc_info.value)

    @patch.object(MultimodalVLMModelslimV1QuantConfig, "from_base")
    def test_quantize_normal_flow_calls_quant_process(self, mock_from_base):
        """验证量化主入口在校验通过时会正确调用 `quant_process`。"""
        converted_cfg = Mock()
        mock_from_base.return_value = converted_cfg

        self.service.quant_process = Mock(return_value="ok")

        base_cfg = Mock(spec=BaseQuantConfig)
        result = self.service.quantize(
            base_cfg,
            self.model_adapter,
            self.save_path,
            device=self.device,
            device_indices=[0],
        )

        assert result == "ok"
        mock_from_base.assert_called_once_with(base_cfg)
        self.service.quant_process.assert_called_once_with(
            converted_cfg,
            self.model_adapter,
            self.save_path,
            self.device,
            [0],
        )

    @pytest.mark.parametrize(
        "runner_value, expected_type",
        [
            (RunnerType.MODEL_WISE, RunnerType.MODEL_WISE),
            (RunnerType.LAYER_WISE, RunnerType.LAYER_WISE),
            ("unknown", RunnerType.LAYER_WISE),
        ],
    )
    @patch("msmodelslim.core.quant_service.multimodal_vlm_v1.quant_service.get_logger")
    def test_choose_runner_type_with_logging(self, mock_get_logger, runner_value, expected_type):
        """覆盖 `_choose_runner_type` 的所有分支，同时验证日志打印行为。"""
        quant_cfg = MagicMock()
        quant_cfg.spec.runner = runner_value

        adapter = Mock(spec=PipelineInterface)
        result = self.service._choose_runner_type(quant_cfg, adapter)

        assert result == expected_type
        # 至少会调用一次日志接口，确保路径被真实执行
        assert mock_get_logger.return_value.info.called

    @patch("msmodelslim.core.quant_service.multimodal_vlm_v1.quant_service.get_logger")
    @patch("msmodelslim.core.quant_service.multimodal_vlm_v1.quant_service.LayerWiseRunner")
    @patch("msmodelslim.core.quant_service.multimodal_vlm_v1.quant_service.torch")
    @patch("msmodelslim.core.quant_service.multimodal_vlm_v1.quant_service.seed_all")
    def test_quant_process_full_flow_with_save_path_and_npu(
        self,
        mock_seed_all,
        mock_torch,
        mock_runner_cls,
        mock_get_logger,
    ):
        """
        完整流程测试（带 save_path & NPU 设备）：
        - 校验随机种子设置
        - 校验 NPU 编译模式配置
        - 校验数据集加载与 default_text 赋值
        - 校验持久化配置与 Processor 注册
        - 校验最终的 runner.run 调用参数
        """
        # 准备 quant_config.spec 的各项字段
        process_cfgs: List[Mock] = [Mock(), Mock()]
        save_cfgs: List[Mock] = [Mock(), Mock()]

        spec = Mock()
        spec.dataset = "mock_dataset_path"
        spec.default_text = "default prompt"
        spec.process = process_cfgs
        spec.save = save_cfgs
        spec.runner = "layer_wise"

        quant_cfg = Mock()
        quant_cfg.spec = spec

        # LayerWiseRunner 返回的 runner mock
        runner = Mock()
        mock_runner_cls.return_value = runner

        # 调用被测方法
        result = self.service.quant_process(
            quant_config=quant_cfg,
            model_adapter=self.model_adapter,
            save_path=self.save_path,
            device=self.device,
            device_indices=[0, 1],
        )

        # quant_process 不需要返回值，这里只确认流程顺利结束
        assert result is None

        # 随机种子设置
        mock_seed_all.assert_called_once()

        # NPU 编译模式设置
        mock_torch.npu.set_compile_mode.assert_called_once_with(jit_compile=False)

        # 数据集加载与 default_text 设置
        assert self.dataset_loader.default_text == "default prompt"
        self.dataset_loader.get_dataset_by_name.assert_called_once_with("mock_dataset_path")

        # save_path 有效时，save_cfg 需要被更新目录，并附加到最终 process 列表
        for save_cfg in save_cfgs:
            save_cfg.set_save_directory.assert_called_once_with(self.save_path)

        # Runner 创建与 Processor 注册
        mock_runner_cls.assert_called_once_with(adapter=self.model_adapter)
        expected_calls = [call(processor_cfg=cfg) for cfg in process_cfgs + save_cfgs]
        runner.add_processor.assert_has_calls(expected_calls, any_order=False)

        # 最终量化执行
        runner.run.assert_called_once_with(calib_data="mock_dataset", device=self.device)
        # 确保结束日志被打印
        assert mock_get_logger.return_value.info.called

    @patch("msmodelslim.core.quant_service.multimodal_vlm_v1.quant_service.get_logger")
    @patch("msmodelslim.core.quant_service.multimodal_vlm_v1.quant_service.LayerWiseRunner")
    @patch("msmodelslim.core.quant_service.multimodal_vlm_v1.quant_service.seed_all")
    def test_quant_process_without_save_path_and_non_layerwise_runner(
        self,
        mock_seed_all,
        mock_runner_cls,
        mock_get_logger,
    ):
        """
        测试以下组合场景：
        - device 为 CPU（跳过 NPU 特有逻辑）
        - runner 类型不是 "layer_wise"（触发 warning 日志）
        - save_path 为 None（不注册 save processor）
        """
        process_cfgs: List[Mock] = [Mock()]
        save_cfgs: List[Mock] = [Mock()]

        spec = Mock()
        spec.dataset = "cpu_dataset"
        spec.default_text = "cpu prompt"
        spec.process = process_cfgs
        spec.save = save_cfgs
        spec.runner = "not-layer-wise"

        quant_cfg = Mock()
        quant_cfg.spec = spec

        runner = Mock()
        mock_runner_cls.return_value = runner
        cpu_device = DeviceType.CPU

        # 注意：这里不提供 save_path
        self.service.quant_process(
            quant_config=quant_cfg,
            model_adapter=self.model_adapter,
            save_path=None,
            device=cpu_device,
            device_indices=None,
        )

        # CPU 场景下，seed_all 仍应调用
        mock_seed_all.assert_called_once()

        # save_path 为 None，save_cfg 不应触发 set_save_directory
        for save_cfg in save_cfgs:
            save_cfg.set_save_directory.assert_not_called()

        # Runner 仍然需要创建并执行
        mock_runner_cls.assert_called_once_with(adapter=self.model_adapter)
        runner.add_processor.assert_has_calls(
            [call(processor_cfg=cfg) for cfg in process_cfgs],
            any_order=False,
        )
        runner.run.assert_called_once_with(calib_data="mock_dataset", device=cpu_device)

        # 非 layer_wise 的 runner 应触发 warning 日志
        mock_logger = mock_get_logger.return_value
        mock_logger.warning.assert_called_once()
