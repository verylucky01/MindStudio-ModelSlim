# -*- coding: utf-8 -*-
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
from typing import Any, List
from unittest.mock import MagicMock, patch

# 修复 torch 导入警告，假设测试环境中 torch 已安装
try:
    import torch
    from torch import nn
except ImportError:
    torch = None
    nn = None

from msmodelslim.processor.anti_outlier.smooth_base import BaseSmoothProcessor
from msmodelslim.processor.anti_outlier.iter_smooth import IterSmoothProcessorConfig
from msmodelslim.core.graph.adapter_types import AdapterConfig
from msmodelslim.core.base.protocol import BatchProcessRequest
from msmodelslim.utils.exception import SchemaValidateError


class ConcreteSmoothProcessor(BaseSmoothProcessor):
    """用于测试的具体实现类，实现了 BaseSmoothProcessor 的抽象方法"""

    def apply_smooth_algorithm(self, subgraph_obj: Any, linear_names: List[str]) -> None:
        """测试用的平滑算法应用，不做任何操作"""
        pass

    def _validate_adapter_interface(self, adapter: object) -> None:
        """测试用的适配器接口验证，不做任何验证"""
        pass


class TestBaseSmoothProcessor(unittest.TestCase):
    def setUp(self):
        self.model = MagicMock(spec=nn.Module)
        self.model.config = MagicMock()
        self.model.config.num_attention_heads = 8
        self.model.config.num_key_value_heads = 4
        self.config = IterSmoothProcessorConfig(
            alpha=0.5,
            symmetric=True,
            enable_subgraph_type=["norm-linear", "linear-linear", "ov", "up-down"],
            include=["*layer1*"],
            exclude=["*layer3*"]
        )
        self.adapter = MagicMock()
        self.processor = ConcreteSmoothProcessor(self.model, self.config, self.adapter)

    def test_validate_parameters_valid(self):
        self.processor._validate_parameters()
        # No exception should be raised
        self.assertTrue(True)

    def test_validate_parameters_invalid_subgraph_type(self):
        invalid_config = IterSmoothProcessorConfig(
            enable_subgraph_type=["invalid-type"]
        )
        processor = ConcreteSmoothProcessor(self.model, invalid_config)
        with self.assertRaises(SchemaValidateError):
            processor._validate_parameters()

    def test_get_num_key_value_heads(self):
        self.model.config.num_key_value_heads = 4
        result = self.processor._get_num_key_value_heads()
        self.assertEqual(result, 4)

    def test_is_data_free(self):
        result = self.processor.is_data_free()
        self.assertFalse(result)

    def test_pre_run(self):
        self.adapter.get_adapter_config_for_subgraph.return_value = ["config1", "config2"]
        self.processor.pre_run()
        self.assertEqual(self.processor.global_adapter_config, ["config1", "config2"])

    def test_preprocess(self):
        request = MagicMock(spec=BatchProcessRequest)
        request.name = "test_module"
        request.module = MagicMock(spec=nn.Module)
        self.processor.global_adapter_config = ["config1", "config2"]
        mock_return_value = ["filtered_config"]
        with patch.object(
                self.processor, '_filter_adapter_configs_by_config', return_value=mock_return_value
        ) as mock_filter:
            with patch.object(self.processor, '_install_statistics_hooks') as mock_install:
                self.processor.preprocess(request)
                mock_filter.assert_called_once_with(["config1", "config2"], self.config, "test_module")
                self.assertEqual(self.processor.adapter_config, ["filtered_config"])
                mock_install.assert_called_once()

    def test_postprocess(self):
        request = MagicMock(spec=BatchProcessRequest)
        request.name = "test_module"
        request.module = MagicMock(spec=nn.Module)

        # 设置 stats_collector
        mock_stats_collector = MagicMock()
        self.processor.stats_collector = mock_stats_collector

        with patch.object(self.processor, '_process_subgraphs_by_priority') as mock_process:
            with patch.object(self.processor, '_remove_all_hooks') as mock_remove:
                self.processor.postprocess(request)
                mock_process.assert_called_once()
                mock_stats_collector.clear_stats.assert_called_once()
                mock_remove.assert_called_once()

    def test_filter_adapter_configs_by_config(self):
        adapter_configs = [
            MagicMock(spec=AdapterConfig, subgraph_type="norm-linear", mapping=MagicMock(source="layer1.module")),
            MagicMock(spec=AdapterConfig, subgraph_type="linear-linear", mapping=MagicMock(source="layer2.module")),
            MagicMock(spec=AdapterConfig, subgraph_type="ov", mapping=MagicMock(source="layer3.module")),
            MagicMock(spec=AdapterConfig, subgraph_type="up-down", mapping=MagicMock(source="layer4.module"))
        ]
        result = self.processor._filter_adapter_configs_by_config(adapter_configs, self.config, "layer1")
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].mapping.source, "layer1.module")

    def test_install_statistics_hooks(self):
        # 设置 stats_collector
        mock_stats_collector = MagicMock()
        mock_stats_collector.create_hook = MagicMock(return_value=lambda x, y, z: None)
        self.processor.stats_collector = mock_stats_collector

        self.processor.adapter_config = [
            MagicMock(
                spec=AdapterConfig,
                subgraph_type="norm-linear",
                mapping=MagicMock(targets=["target1", "target2"])
            )
        ]

        with patch.object(self.processor.hook_manager, 'install_hook') as mock_install:
            self.processor._install_statistics_hooks()
            # 应该为每个 adapter_config 安装一个 hook（使用 targets[0]）
            mock_install.assert_called_once()
            # 验证 create_hook 被调用
            mock_stats_collector.create_hook.assert_called_once()

    def test_remove_all_hooks(self):
        self.processor.hook_manager.hook_handles = {
            "module1": MagicMock(),
            "module2": MagicMock()
        }
        with patch.object(self.processor.hook_manager.hook_handles["module1"], 'remove') as mock_remove1:
            with patch.object(self.processor.hook_manager.hook_handles["module2"], 'remove') as mock_remove2:
                self.processor._remove_all_hooks()
                mock_remove1.assert_called_once()
                mock_remove2.assert_called_once()
                self.assertEqual(len(self.processor.hook_manager.hook_handles), 0)

    def test_apply_norm_linear_smooth(self):
        adapter_config = MagicMock(
            spec=AdapterConfig,
            mapping=MagicMock(source="source", targets=["target1", "target2"])
        )
        source_module = MagicMock(spec=nn.Module)
        target_modules = [MagicMock(spec=nn.Module), MagicMock(spec=nn.Module)]

        def get_submodule_mock(x):
            if x == "source":
                return source_module
            elif x == "target1":
                return target_modules[0]
            else:
                return target_modules[1]

        self.model.get_submodule.side_effect = get_submodule_mock
        with patch.object(self.processor, 'apply_smooth_algorithm') as mock_apply:
            self.processor._apply_norm_linear_smooth(adapter_config)
            mock_apply.assert_called_once()

    def test_apply_linear_linear_smooth(self):
        adapter_config = MagicMock(spec=AdapterConfig, mapping=MagicMock(source="source", targets=["target1"]))
        source_module = MagicMock(spec=nn.Module)
        target_module = MagicMock(spec=nn.Module)

        def get_submodule_for_linear_linear(x):
            return source_module if x == "source" else target_module

        self.model.get_submodule.side_effect = get_submodule_for_linear_linear
        with patch.object(self.processor, 'apply_smooth_algorithm') as mock_apply:
            self.processor._apply_linear_linear_smooth(adapter_config)
            mock_apply.assert_called_once()

    def test_apply_ov_smooth_with_fusion(self):
        adapter_config = MagicMock(spec=AdapterConfig, fusion=MagicMock(fusion_type="qkv"))
        with patch.object(self.processor, '_apply_qkv_fusion_smooth') as mock_fusion:
            with patch.object(self.processor, '_apply_standard_ov_smooth') as mock_standard:
                self.processor._apply_ov_smooth(adapter_config)
                mock_fusion.assert_called_once_with(adapter_config)
                mock_standard.assert_not_called()

    def test_apply_ov_smooth_standard(self):
        adapter_config = MagicMock(spec=AdapterConfig, fusion=None)
        with patch.object(self.processor, '_apply_qkv_fusion_smooth') as mock_fusion:
            with patch.object(self.processor, '_apply_standard_ov_smooth') as mock_standard:
                self.processor._apply_ov_smooth(adapter_config)
                mock_fusion.assert_not_called()
                mock_standard.assert_called_once_with(adapter_config)

    def test_apply_qkv_fusion_smooth_qkv(self):
        adapter_config = MagicMock(
            spec=AdapterConfig,
            mapping=MagicMock(source="v_module", targets=["o_module"]),
            fusion=MagicMock(
                fusion_type="qkv",
                num_attention_heads=8,
                num_key_value_heads=4
            )
        )
        v_module = MagicMock(spec=nn.Linear)
        o_module = MagicMock(spec=nn.Module)

        def get_submodule_for_qkv(x):
            return v_module if x == "v_module" else o_module

        self.model.get_submodule.side_effect = get_submodule_for_qkv
        mock_path = 'msmodelslim.processor.anti_outlier.smooth_base.VirtualVModuleFromQKVFused'
        with patch(mock_path) as mock_virtual:
            mock_virtual_instance = MagicMock()
            mock_virtual.return_value = mock_virtual_instance
            with patch.object(self.processor, 'apply_smooth_algorithm') as mock_apply:
                self.processor._apply_qkv_fusion_smooth(adapter_config)
                mock_virtual.assert_called_once()
                mock_apply.assert_called_once()
                mock_virtual_instance.update_weights.assert_called_once()

    def test_apply_standard_ov_smooth(self):
        adapter_config = MagicMock(
            spec=AdapterConfig,
            mapping=MagicMock(source="v_module", targets=["o_module"])
        )
        v_module = MagicMock(spec=nn.Linear)
        o_module = MagicMock(spec=nn.Module)

        def get_submodule_for_standard_ov(x):
            return v_module if x == "v_module" else o_module

        self.model.get_submodule.side_effect = get_submodule_for_standard_ov
        with patch.object(
                self.processor, '_get_num_attention_heads', return_value=8
        ) as mock_heads:
            with patch.object(
                    self.processor, '_get_num_key_value_heads', return_value=4
            ) as mock_kv_heads:
                with patch.object(self.processor, 'apply_smooth_algorithm') as mock_apply:
                    self.processor._apply_standard_ov_smooth(adapter_config)
                    mock_heads.assert_called_once()
                    mock_kv_heads.assert_called_once()
                    mock_apply.assert_called_once()

    def test_apply_up_down_smooth(self):
        adapter_config = MagicMock(
            spec=AdapterConfig,
            mapping=MagicMock(source="up_module", targets=["down_module", "gate_module"])
        )
        up_module = MagicMock(spec=nn.Module)
        down_module = MagicMock(spec=nn.Module)
        gate_module = MagicMock(spec=nn.Module)

        def get_submodule_for_updown(x):
            if x == "up_module":
                return up_module
            elif x == "down_module":
                return down_module
            else:
                return gate_module

        self.model.get_submodule.side_effect = get_submodule_for_updown
        with patch.object(self.processor, 'apply_smooth_algorithm') as mock_apply:
            self.processor._apply_up_down_smooth(adapter_config)
            mock_apply.assert_called_once()

    def test_process_single_subgraph(self):
        adapter_config = MagicMock(spec=AdapterConfig, subgraph_type="norm-linear")
        adapter_config.mapping = MagicMock(source="source")
        with patch.object(self.processor, '_apply_norm_linear_smooth') as mock_handler:
            self.processor._process_single_subgraph(adapter_config)
            mock_handler.assert_called_once_with(adapter_config)

    def test_virtual_module_creation_kv_error(self):
        adapter_config = MagicMock(
            spec=AdapterConfig,
            mapping=MagicMock(source="v_module", targets=["o_module"]),
            fusion=MagicMock(
                fusion_type="kv",
                num_attention_heads=8,
                custom_config={"qk_nope_head_dim": 64, "v_head_dim": 64}
            )
        )
        v_module = MagicMock(spec=nn.Linear)
        o_module = MagicMock(spec=nn.Module)

        def get_submodule_for_kv_error(x):
            return v_module if x == "v_module" else o_module

        self.model.get_submodule.side_effect = get_submodule_for_kv_error
        mock_kv_path = 'msmodelslim.processor.anti_outlier.smooth_base.VirtualVModuleFromKVFused'
        with patch(mock_kv_path) as mock_virtual:
            mock_virtual.side_effect = Exception("Failed to create virtual module")
            with self.assertRaises(Exception):
                self.processor._apply_qkv_fusion_smooth(adapter_config)

    def test_process_subgraphs_by_priority(self):
        self.processor.adapter_config = [
            MagicMock(spec=AdapterConfig, subgraph_type="up-down", mapping=MagicMock(source="up"),
                      priority=1),
            MagicMock(spec=AdapterConfig, subgraph_type="ov", mapping=MagicMock(source="ov"),
                      priority=2),
            MagicMock(spec=AdapterConfig, subgraph_type="norm-linear", mapping=MagicMock(source="norm"),
                      priority=3),
            MagicMock(spec=AdapterConfig, subgraph_type="linear-linear", mapping=MagicMock(source="linear"),
                      priority=4)
        ]
        with patch.object(self.processor, '_process_single_subgraph') as mock_process:
            self.processor._process_subgraphs_by_priority()
            self.assertEqual(mock_process.call_count, 4)
            calls = mock_process.call_args_list
            self.assertEqual(calls[0][0][0].subgraph_type, "up-down")
            self.assertEqual(calls[1][0][0].subgraph_type, "ov")
            self.assertEqual(calls[2][0][0].subgraph_type, "norm-linear")
            self.assertEqual(calls[3][0][0].subgraph_type, "linear-linear")


if __name__ == '__main__':
    unittest.main()
