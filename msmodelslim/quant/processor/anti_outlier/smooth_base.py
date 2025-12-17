#  -*- coding: utf-8 -*-
#  Copyright (c) 2024-2024 Huawei Technologies Co., Ltd.
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

from abc import ABC, abstractmethod
from enum import Enum
from typing import List, Optional, Dict, Any, Literal

from torch import nn

from msmodelslim.core.QAL.qtypes import (
    LinearLinearSubgraph,
    NormLinearSubgraph,
    UpDownSubgraph,
    OVSubgraph
)
from msmodelslim.core.base.protocol import BatchProcessRequest
from msmodelslim.core.graph.adapter_types import AdapterConfig
from msmodelslim.quant.processor.base import AutoSessionProcessor, AutoProcessorConfig
from msmodelslim.utils.config_map import ConfigSet
from msmodelslim.utils.exception import MisbehaviorError, SchemaValidateError, UnsupportedError
from msmodelslim.utils.logging import get_logger

from .common import (
    VirtualVModuleFromQKVFused,
    VirtualVModuleFromKVFused,
    HookManager,
    SubgraphRegistry
)


class BaseSmoothProcessor(AutoSessionProcessor, ABC):
    """
    Base Smooth Processor

    Responsibilities:
    1. Provide processing flow framework (template method)
    2. Define abstract interfaces (implemented by subclasses)
    3. Manage common resources (config, hooks, context)
    """
    SUBGRAPH_HANDLERS = SubgraphRegistry.NAME_TO_HANDLER
    SUBGRAPH_PRIORITY = {
        "up-down": 1,
        "ov": 2,
        "norm-linear": 3,
        "linear-linear": 4
    }

    def __init__(self, model: nn.Module, config: AutoProcessorConfig, adapter: Optional[Any] = None):
        super().__init__(model)
        self.model = model
        self._validate_adapter_interface(adapter)
        self.adapter = adapter
        self.config = config
        self.hook_manager = HookManager(model)
        self.context_builder = None
        self.global_adapter_config = None
        self.adapter_config = None
        self.stats_collector = None

    @abstractmethod
    def apply_smooth_algorithm(self, subgraph_obj: Any, linear_names: List[str]) -> None:
        """Apply smooth algorithm (must be implemented by subclass)"""
        ...

    @abstractmethod
    def _validate_adapter_interface(self, adapter: object) -> None:
        """Validate if the adapter implements the required interface.
           Raise UnsupportedError if the adapter does not implement the required interface.
        """
        ...

    def pre_run(self) -> None:
        self.global_adapter_config = self.adapter.get_adapter_config_for_subgraph()
        self._validate_parameters()

    def preprocess(self, request: BatchProcessRequest) -> None:
        self.adapter_config = self._filter_adapter_configs_by_config(
            self.global_adapter_config,
            self.config,
            request.name
        )
        get_logger().debug(
            "Processed %d subgraphs for submodule %s",
            len(self.adapter_config), request.name
        )
        self._install_statistics_hooks()

    def postprocess(self, request: BatchProcessRequest) -> None:
        self._process_subgraphs_by_priority()

        # Cleanup resources
        if self.stats_collector:
            self.stats_collector.clear_stats()
        self._remove_all_hooks()
        get_logger().debug("Completed smoothing, cleared statistics and hooks")

    def _validate_parameters(self) -> None:
        """Validate all parameter legality"""
        valid_types = SubgraphRegistry.get_all_supported_types()
        for subgraph_type in self.config.enable_subgraph_type:
            if not SubgraphRegistry.is_supported(subgraph_type):
                raise SchemaValidateError(
                    f"Elements in enable_subgraph_type must be in {valid_types}, "
                    f"current element: {subgraph_type}",
                    action=f"Please use only valid subgraph types: {valid_types}")

    def _filter_adapter_configs_by_config(
            self,
            adapter_configs: List[AdapterConfig],
            config: AutoProcessorConfig,
            scope: str
    ) -> List[AdapterConfig]:
        """Filter adapter configurations based on config"""
        result = []
        layer_prefix = f"{scope}." if scope != "" else ""
        include = ConfigSet(config.include) if config.include else ConfigSet(["*"])
        exclude = ConfigSet(config.exclude) if config.exclude else ConfigSet([])

        for adapter_config in adapter_configs:
            if adapter_config.subgraph_type not in config.enable_subgraph_type:
                continue
            if not adapter_config.mapping:
                continue

            source_name = adapter_config.mapping.source
            if not source_name.startswith(layer_prefix):
                continue
            if source_name not in include:
                continue
            if source_name in exclude:
                continue

            result.append(adapter_config)

        return result

    def _install_statistics_hooks(self) -> None:
        """Install statistics hooks for all subgraph targets"""
        for adapter_config in self.adapter_config:
            # 只收集targets中第一个层的激活
            target_name = adapter_config.mapping.targets[0]
            get_logger().debug("Installing statistics hooks for %s", target_name)
            hook_fn = self.stats_collector.create_hook(target_name, adapter_config.subgraph_type)
            self.hook_manager.install_hook(target_name, hook_fn, adapter_config.subgraph_type)

    def _process_subgraphs_by_priority(self) -> None:
        """Process subgraphs in priority order"""
        get_logger().debug("Starting smoothing application")
        sorted_configs = sorted(
            self.adapter_config,
            key=lambda x: self.SUBGRAPH_PRIORITY.get(x.subgraph_type, 999)
        )

        for idx, adapter_config in enumerate(sorted_configs, start=1):
            priority = self.SUBGRAPH_PRIORITY.get(adapter_config.subgraph_type, 999)
            get_logger().debug(
                "  %d. %s (priority: %d) - %s",
                idx, adapter_config.subgraph_type, priority, adapter_config.mapping.source
            )

            try:
                self._process_single_subgraph(adapter_config)
            except Exception as e:
                get_logger().error(
                    "Error processing subgraph %s: %s",
                    adapter_config.mapping.source, e
                )

    def _process_single_subgraph(self, adapter_config: AdapterConfig) -> None:
        """Process single subgraph (dispatch to specific handler)"""
        handler_name = SubgraphRegistry.get_handler_name(adapter_config.subgraph_type)
        handler = getattr(self, handler_name, None)

        get_logger().debug(
            "    Mapping: %s -> %s",
            adapter_config.mapping.source, adapter_config.mapping.targets
        )
        handler(adapter_config)

    def _apply_up_down_smooth(self, adapter_config: AdapterConfig) -> None:
        """Apply Up-Down smoothing (Priority 1)"""
        up_name = adapter_config.mapping.source
        up_module = self.model.get_submodule(up_name)
        down_name = adapter_config.mapping.targets[0] if adapter_config.mapping.targets else ""
        down_module = self.model.get_submodule(down_name) if down_name else None

        if not up_module or not down_module:
            get_logger().warning("Failed to get modules for up-down subgraph: %s", adapter_config.mapping.source)
            return
        gate_module = None
        get_logger().debug(
            "Up module name:%s Down module name:%s",
            up_name,
            down_name
        )
        self.apply_smooth_algorithm(
            UpDownSubgraph(up_module, down_module, gate_module),
            [down_name]
        )

    def _apply_ov_smooth(self, adapter_config: AdapterConfig) -> None:
        """Apply OV smoothing (Priority 2)"""
        fusion = adapter_config.fusion
        fusion_flag = fusion is not None and fusion.fusion_type != "none"

        try:
            if fusion_flag:
                self._apply_qkv_fusion_smooth(adapter_config)
            else:
                self._apply_standard_ov_smooth(adapter_config)
        except Exception as e:
            get_logger().error("Error occurred while applying OV smoothing: %s", e)

    def _apply_qkv_fusion_smooth(self, adapter_config: AdapterConfig) -> None:
        """Apply QKV fusion smoothing (OV sub-method)"""
        v_name = adapter_config.mapping.source
        o_name = adapter_config.mapping.targets[0] if adapter_config.mapping.targets else ''
        v_module = self.model.get_submodule(v_name)
        o_module = self.model.get_submodule(o_name)
        fusion = adapter_config.fusion
        extra_config = getattr(adapter_config, 'extra_config', None)

        if not isinstance(v_module, nn.Linear):
            get_logger().warning("V module %s is not Linear type, skipping QKV fusion", v_name)
            return
        if not o_module:
            get_logger().warning("O module %s not found, skipping QKV fusion", o_name)
            return

        # Create virtual V module
        if fusion.fusion_type == "kv":
            virtual_v_module = VirtualVModuleFromKVFused(
                v_module,
                num_attention_heads=fusion.num_attention_heads,
                qk_nope_head_dim=fusion.custom_config["qk_nope_head_dim"],
                v_head_dim=fusion.custom_config["v_head_dim"]
            )
        elif fusion.fusion_type == "qkv":
            virtual_v_module = VirtualVModuleFromQKVFused(
                v_module,
                num_attention_heads=fusion.num_attention_heads,
                num_key_value_heads=fusion.num_key_value_heads
            )
        else:
            raise UnsupportedError(f"Unsupported fusion type: {fusion.fusion_type}")

        self.apply_smooth_algorithm(
            OVSubgraph(
                o_proj=o_module,
                v_proj=virtual_v_module,
                num_attention_heads=fusion.num_attention_heads,
                key_value_heads=fusion.num_key_value_heads,
                extra_config=extra_config
            ),
            [o_name]
        )
        virtual_v_module.update_weights()
        get_logger().debug("Successfully applied QKV fusion smoothing: %s -> %s", v_name, o_name)

    def _apply_standard_ov_smooth(self, adapter_config: AdapterConfig) -> None:
        """Apply standard OV smoothing (OV sub-method)"""
        v_name = adapter_config.mapping.source
        o_name = adapter_config.mapping.targets[0] if adapter_config.mapping.targets else ''
        v_module = self.model.get_submodule(v_name)
        o_module = self.model.get_submodule(o_name)
        extra_config = getattr(adapter_config, 'extra_config', None)

        if not isinstance(v_module, nn.Linear):
            get_logger().warning("V module %s is not Linear type, skipping standard OV smoothing", v_name)
            return
        if not o_module:
            get_logger().warning("O module %s not found, skipping standard OV smoothing", o_name)
            return

        num_attention_heads = self._get_num_attention_heads()
        num_key_value_heads = self._get_num_key_value_heads()

        self.apply_smooth_algorithm(
            OVSubgraph(
                o_proj=o_module,
                v_proj=v_module,
                num_attention_heads=num_attention_heads,
                key_value_heads=num_key_value_heads,
                extra_config=extra_config
            ),
            [o_name]
        )
        get_logger().debug("Successfully applied standard OV smoothing: %s -> %s", v_name, o_name)

    def _apply_norm_linear_smooth(self, adapter_config: AdapterConfig) -> None:
        """Apply Norm-Linear smoothing (Priority 3)"""
        source_module = self.model.get_submodule(adapter_config.mapping.source)
        target_modules = [self.model.get_submodule(name) for name in adapter_config.mapping.targets]
        target_modules = [m for m in target_modules if m is not None]
        target_names = adapter_config.mapping.targets

        if not source_module or not target_modules:
            get_logger().warning("Failed to get modules for norm-linear subgraph: %s", adapter_config.mapping.source)
            return

        self.apply_smooth_algorithm(
            NormLinearSubgraph(source_module, target_modules),
            target_names
        )

    def _apply_linear_linear_smooth(self, adapter_config: AdapterConfig) -> None:
        """Apply Linear-Linear smoothing (Priority 4)"""
        source_module = self.model.get_submodule(adapter_config.mapping.source)
        target_modules = [self.model.get_submodule(name) for name in adapter_config.mapping.targets]
        target_modules = [m for m in target_modules if m is not None]
        target_names = adapter_config.mapping.targets

        if not source_module or not target_modules:
            get_logger().warning("Failed to get modules for linear-linear subgraph: %s", adapter_config.mapping.source)
            return

        target_module = target_modules[0]
        self.apply_smooth_algorithm(
            LinearLinearSubgraph(source_module, target_module),
            [target_names]
        )

    def _remove_all_hooks(self) -> None:
        """Remove all installed hooks"""
        self.hook_manager.remove_all_hooks()

    def _get_num_attention_heads(self) -> int:
        """Get number of attention heads from model config"""
        num_attention_heads = None
        key_attention_heads = ["num_attention_heads", "n_head", "num_heads", "heads_num"]
        for key in key_attention_heads:
            if hasattr(self.model.config, key):
                num_attention_heads = getattr(self.model.config, key)
                break

        if not num_attention_heads:
            raise MisbehaviorError(
                "the config of model must have num_attention_heads, n_head or num_heads, "
                "please check or modify the config file"
            )
        return num_attention_heads

    def _get_num_key_value_heads(self) -> int:
        """Get number of key-value heads from model config"""
        if not hasattr(self.model.config, "num_key_value_heads"):
            num_key_value_heads = self._get_num_attention_heads()
            get_logger().warning(
                "Failed to obtain `num_key_value_heads`, assuming Multi-head Attention by default."
            )
        else:
            num_key_value_heads = self.model.config.num_key_value_heads

        if not num_key_value_heads:
            raise MisbehaviorError(
                "the config of model must have num_key_value_heads, "
                "please check or modify the config file"
            )
        return num_key_value_heads

