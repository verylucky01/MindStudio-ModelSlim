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

from typing import Annotated, List, Optional, Any, Literal

from pydantic import AfterValidator, Field
from torch import nn
from msmodelslim.core.base.protocol import BatchProcessRequest
from msmodelslim.core.quantizer.base import QConfig
from msmodelslim.ir.qal.qregistry import QABCRegistry
from msmodelslim.processor.base import AutoProcessorConfig, AutoSessionProcessor
from msmodelslim.utils.logging import get_logger
from msmodelslim.utils.exception import UnsupportedError
from msmodelslim.processor.anti_outlier.smooth_base import BaseSmoothProcessor
from msmodelslim.core.graph.adapter_types import MappingConfig
from msmodelslim.utils.validation.pydantic import is_string_list
from .awq_stats_collector import AWQStatsCollector
from .interface import AWQInterface
from .api import awq
from .common import AWQConfig, AWQContext
from .best_scales_search import AWQBestScalesSearcher

class AWQProcessorConfig(AutoProcessorConfig):
    type: Literal["awq"] = "awq"
    weight_qconfig: QConfig
    enable_subgraph_type: Annotated[list, AfterValidator(is_string_list)] = Field(
        default_factory=lambda: ["norm-linear", "linear-linear", "ov", "up-down"]
    )
    n_grid: Annotated[int, Field(gt=0)] = 20
    include: Optional[List[str]] = None
    exclude: Optional[List[str]] = None

    
@QABCRegistry.register(dispatch_key=AWQProcessorConfig, abc_class=AutoSessionProcessor)
class AWQProcessor(BaseSmoothProcessor):
    """
    AWQ (Activation-aware Weight Quantization) Processor
    """
    def __init__(self, model: nn.Module, config: AWQProcessorConfig, adapter: Any):
        super().__init__(model, config, adapter)
        self.stats_collector = AWQStatsCollector(model)
        self.awq_config = AWQConfig(
            version=1,
            awq_searcher=AWQBestScalesSearcher(
                weight_qconfig=config.weight_qconfig,
                n_grid=config.n_grid,
            ),
        )
    
    def _validate_adapter_interface(self, adapter: Any) -> None:
        if not isinstance(adapter, AWQInterface):
            raise UnsupportedError(
                f"{adapter.__class__.__name__} does not implement AWQInterface",
                action=f"Please ensure {adapter.__class__.__name__} inherits from AWQInterface "
                f"and implements get_adapter_config_for_subgraph()",
            )
    
    def pre_run(self) -> None:
        self.global_adapter_config = self.adapter.get_adapter_config_for_subgraph()
        self._validate_parameters()
    
    def support_distributed(self) -> bool:
        return False
    
    def preprocess(self, request: BatchProcessRequest) -> None:
        if self.global_adapter_config is None:
            get_logger().warning("No global adapter config found for AWQProcessor, skipping preprocessing")
            return

        self.adapter_config = self._filter_adapter_configs_by_config(
            self.global_adapter_config,
            self.config,
            request.name
        )
        get_logger().debug(
            "Processed %d subgraphs for submodule %s",
            len(self.adapter_config), request.name
        )

        for adapter_config in self.adapter_config:
            self._install_hooks(adapter_config.mapping)

    def _install_hooks(self, mapping_config: MappingConfig) -> None:
        """Install activation-mean and kwargs hooks for a single subgraph."""
        target_names = mapping_config.targets
        if target_names is not None:
            self.stats_collector.observe_activation(target_names[0])
        ancestor_name = self._find_lowest_common_ancestor(target_names)
        if ancestor_name is not None:
            self.stats_collector.observe_kwargs(ancestor_name)
    
    def _find_lowest_common_ancestor(self, target_names: List[str]) -> Optional[str]:
        """Find the lowest common ancestor module name for a list of target module names.
        Examples:
            ["model.layers.0.self_attn.q_proj", "model.layers.0.self_attn.k_proj"]
            -> "model.layers.0.self_attn"

            ["model.layers.0.mlp.gate_proj", "model.layers.0.mlp.up_proj"]
            -> "model.layers.0.mlp"
            
            ["model.layers.0.mlp.down_proj"]
            -> "model.layers.0.mlp.down_proj" (single target, ancestor is itself)

            ["a.b.c", "x.y.z"]
            -> None
        """
        if not target_names:
            return None

        if len(target_names) == 1:
            return target_names[0]
        
        split_names = [name.split(".") for name in target_names]
        mid_depth = min(len(parts) for parts in split_names)
        
        common_parts = []
        for i in range(mid_depth):
            segment = split_names[0][i]
            if all(parts[i] == segment for parts in split_names):
                common_parts.append(segment)
            else:
                break
        
        if not common_parts:
            return None
        return ".".join(common_parts)

    def postprocess(self, request: BatchProcessRequest) -> None:
        self.stats_collector.stop_observing()

        self._process_subgraphs_by_priority()
        
        self.stats_collector.clear_stats()
        self.adapter_config = None

    def apply_smooth_algorithm(self, subgraph_obj: Any, linear_names: List[str]) -> None:
        target_name = linear_names[0] if linear_names else None
        if target_name is None:
            get_logger().warning("No target names for subgraph, skipping")
            return

        ancestor_name = self._find_lowest_common_ancestor(linear_names)
        if ancestor_name is None:
            get_logger().warning("No name found for inspect module of subgraph with target %s, skipping", linear_names)
            return
    
        context = self._build_awq_context(target_name, ancestor_name)
        if context is None:
            get_logger().warning("No statistics for subgraph with target %s, skipping", linear_names)
            return
        awq(subgraph_obj, self.awq_config, context)
        
    def _build_awq_context(
        self, target_name: str, ancestor_name: str,
    ) -> Optional[AWQContext]:
        act_mean = self.stats_collector.get_activation_mean(target_name)
        if act_mean is None:
            get_logger().warning(
                "No activation mean for target module %s, skipping", target_name
            )
            return None
        
        ancestor_args = self.stats_collector.get_block_kwargs(ancestor_name)
        if ancestor_args is None or len(ancestor_args) == 0:
            get_logger().warning(
                "No kwargs cache for parent module %s, skipping", ancestor_name
            )
            return None
        
        ancestor_module = self._resolve_module(ancestor_name)
        if ancestor_module is None:
            get_logger().warning(
                "Ancestor module %s not found in model, skipping", ancestor_name
            )
            return None
        
        return AWQContext(
            act_mean=act_mean,
            inspect_module_args=ancestor_args,
            inspect_module=ancestor_module,
        )
        
    def _resolve_module(self, module_name: str) -> Optional[nn.Module]:
        """Safely resolve module by name, return None if not found."""
        try:
            return self.model.get_submodule(module_name)
        except AttributeError:
            get_logger().warning("Module %s not found in model, skipping", module_name)
            return None