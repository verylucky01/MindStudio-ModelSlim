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
from typing import Callable, Dict, List, Optional, Annotated, Literal

import torch
from pydantic import AfterValidator, Field
from torch import nn

from msmodelslim.core.QAL.qregistry import QABCRegistry
from msmodelslim.core.base.protocol import BatchProcessRequest
from msmodelslim.quant.observer.minmax import MinMaxObserverConfig, MsMinMaxObserver
from msmodelslim.quant.processor.base import AutoSessionProcessor, AutoProcessorConfig
from msmodelslim.utils.config_map import ConfigSet
from msmodelslim.utils.exception import ToDoError, UnsupportedError
from msmodelslim.utils.logging import get_logger, logger_setter
from msmodelslim.utils.validation.value import greater_than_zero
from .fused_interface import KVSmoothFusedInterface, KVSmoothFusedType, KVSmoothFusedUnit
from .listener import KVCacheListenerManager


class KeyStatesMaximumCollector:
    def __init__(self):
        self.observers: Dict[int, MsMinMaxObserver] = {}

    def collect(self, layer_idx: int, key_states: torch.Tensor, _: torch.Tensor):
        if layer_idx not in self.observers:
            observer_config = MinMaxObserverConfig(dim=[0, 2], keepdim=False)
            self.observers[layer_idx] = MsMinMaxObserver(observer_config)
            get_logger().debug(f"[KeyStatesMaximumCollector] create new observer for layer {layer_idx}")
        self.observers[layer_idx].update(key_states)
        get_logger().debug(f"[KeyStatesMaximumCollector] update observer for layer {layer_idx}")

    def get(self, layer_idx: int) -> torch.Tensor:
        if layer_idx not in self.observers:
            raise ToDoError(f'Observer for layer {layer_idx} in KeyStatesMaximumCollector not found',
                            action='Please call KeyStatesMaximumCollector.collect first')
        key_mini, key_maxi = self.observers[layer_idx].get_min_max()
        return torch.max(key_mini.abs(), key_maxi.abs())

    def reset(self) -> None:
        self.observers.clear()


class KVSmoothProcessorConfig(AutoProcessorConfig):
    type: Literal["kv_smooth"] = "kv_smooth"
    smooth_factor: Annotated[float, AfterValidator(greater_than_zero)] = 1.0
    include: List[str] = Field(default_factory=lambda: ["*"], description="包含的模块名称")
    exclude: List[str] = Field(default_factory=lambda: [], description="排除的模块名称")


@QABCRegistry.register(dispatch_key=KVSmoothProcessorConfig, abc_class=AutoSessionProcessor)
@logger_setter('msmodelslim.quant.processor.kv_smooth')
class KVSmoothProcessor(AutoSessionProcessor):
    def __init__(
            self,
            model: nn.Module,
            config: KVSmoothProcessorConfig,
            adapter: Optional[object] = None,
    ):
        super().__init__(model)

        if not isinstance(adapter, KVSmoothFusedInterface):
            raise UnsupportedError(f'{adapter.__class__.__name__} does not support KVSmooth',
                                   action='Please provide a valid model adapter '
                                          'which implements KVCacheSmoothFusedInterface')

        self.config: KVSmoothProcessorConfig = config
        self.adapter: KVSmoothFusedInterface = adapter
        self.fused_units_map: Optional[Dict[str, KVSmoothFusedUnit]] = None
        self.listener_manager = KVCacheListenerManager()
        self.collector = KeyStatesMaximumCollector()
        self.include = ConfigSet(config.include) 
        self.exclude = ConfigSet(config.exclude)

    @staticmethod
    def _check_module(full_name: str, submodule: nn.Module, fused_unit: KVSmoothFusedUnit) -> None:
        if not hasattr(submodule, fused_unit.fused_from_key_states_name):
            raise ToDoError(f"attention {full_name} has no submodule {fused_unit.fused_from_key_states_name}",
                            action='Please check the model adapter to '
                                   'ensure the name of the module fused from key states is correct')
        if not hasattr(getattr(submodule, fused_unit.fused_from_key_states_name), 'weight'):
            raise ToDoError(
                f"the module fused from key states "
                f"{full_name}.{fused_unit.fused_from_key_states_name} has no weight",
                action='Please ensure the module is a linear or norm layer')

        if not hasattr(submodule, fused_unit.fused_from_query_states_name):
            raise ToDoError(f"attention {full_name} has no submodule {fused_unit.fused_from_query_states_name}",
                            action='Please check the model adapter to '
                                   'ensure the name of the module fused from query states is correct')
        if not hasattr(getattr(submodule, fused_unit.fused_from_query_states_name), 'weight'):
            raise ToDoError(
                f"the module fused from query states "
                f"{full_name}.{fused_unit.fused_from_query_states_name} has no weight",
                action='Please ensure the module is a linear or norm layer')

    @staticmethod
    def _warning_unmatched_pattern(name: str, config_set: ConfigSet) -> None:
        unmatched_keys = config_set.unmatched_keys()
        unmatched_keys = list(filter(lambda x: x != "*", unmatched_keys))
        if unmatched_keys:
            get_logger().warning(
                f"These {name} patterns are not matched any module, "
                f"please ensure this is as expected: {unmatched_keys}")

    def is_data_free(self) -> bool:
        return False

    def support_distributed(self) -> bool:
        return False

    def need_kv_cache(self) -> bool:
        return True

    def pre_run(self) -> None:
        get_logger().info("Smoothing kvcache")
        get_logger().debug(f"smooth_factor: {self.config.smooth_factor}")
        get_logger().debug(f"include: {self.include}")
        get_logger().debug(f"exclude: {self.exclude}")

        self.fused_units_map = {unit.attention_name: unit for unit in self.adapter.get_kvcache_smooth_fused_subgraph()}
        get_logger().debug(f"fused_subgraph: {self.fused_units_map}")

    def post_run(self) -> None:
        self._warning_unmatched_pattern("include", self.include)
        self._warning_unmatched_pattern("exclude", self.exclude)
        get_logger().info(f"Smooth kvcache success")

    def preprocess(self, request: BatchProcessRequest) -> None:
        get_logger().debug(f"Smoothing kvcache, module: {request.name}")

        def _handle_module(full_name: str, submodule: nn.Module, _: KVSmoothFusedUnit) -> None:
            self.listener_manager.attach_listener_to_module(submodule, listen_helper=self.collector.collect)
            get_logger().debug(f"Attaching stub to module: {full_name}")

        self._match_and_handle(request, _handle_module)

    def postprocess(self, request: BatchProcessRequest) -> None:
        def _handle_module(full_name: str, submodule: nn.Module, fused_unit: KVSmoothFusedUnit) -> None:
            key_abs_max = self.collector.get(fused_unit.layer_idx)

            if fused_unit.fused_type is KVSmoothFusedType.StateViaRopeToLinear:
                self._handle_module_fused_to_linear(full_name, submodule, fused_unit, key_abs_max)
            elif fused_unit.fused_type is KVSmoothFusedType.StateViaRopeToNorm:
                self._handle_module_fused_to_norm(full_name, submodule, fused_unit, key_abs_max)
            else:
                raise UnsupportedError(f"Unsupported kvcache smooth fused type: {fused_unit.fused_type.value}",
                                       action='Please choose a valid fused type')

        self._match_and_handle(request, _handle_module)

        # free resources
        self.listener_manager.remove_listeners()
        self.collector.reset()
        get_logger().debug(f"Smooth kvcache success, module: {request.name}")

    def _match_and_handle(self, request: BatchProcessRequest,
                          func: Callable[[str, nn.Module, KVSmoothFusedUnit], None]) -> None:
        for name, submodule in request.module.named_modules():
            full_name = f"{request.name}.{name}" if request.name != "" else name
            if full_name not in self.fused_units_map:
                continue

            if full_name not in self.include:
                continue

            if full_name in self.exclude:
                continue

            func(full_name, submodule, self.fused_units_map[full_name])

    def _handle_module_fused_to_linear(self, full_name: str, submodule: nn.Module,
                                       fused_unit: KVSmoothFusedUnit, key_abs_max: torch.Tensor) -> None:
        get_logger().debug(f"Smoothing module fused to linear: {full_name}")
        self._check_module(full_name, submodule, fused_unit)

        head_dim = self.adapter.get_head_dim()
        num_key_value_heads = self.adapter.get_num_key_value_heads()
        num_key_value_groups = self.adapter.get_num_key_value_groups()
        get_logger().debug(
            f"head_dim: {head_dim}, "
            f"num_key_value_heads: {num_key_value_heads}, "
            f"num_key_value_groups: {num_key_value_groups}, "
            f"key_abs_max.max: {key_abs_max.max()}, "
            f"key_abs_max.min: {key_abs_max.min()}, "
            f"key_abs_max.mean: {key_abs_max.mean()}")

        scale = key_abs_max.view(num_key_value_heads, 2, head_dim // 2).max(dim=1)[
            0]  # deal with rope: ____ ____ -> ____
        scale = scale.pow(self.config.smooth_factor)  # pow smooth factor
        k_scale = torch.repeat_interleave(scale, repeats=2, dim=0)  # recover rope: ____ -> ____ ____
        q_scale = torch.repeat_interleave(k_scale.view(num_key_value_heads, head_dim), repeats=num_key_value_groups,
                                          dim=0)
        get_logger().debug(f"k_scale: {k_scale}, q_scale: {q_scale}")

        fused_from_key_states_module = getattr(submodule, fused_unit.fused_from_key_states_name)
        fused_from_key_states_module.weight.div_(k_scale.view(-1, 1))
        get_logger().debug(f"weight of module {full_name}.{fused_unit.fused_from_key_states_name} divided")
        if hasattr(fused_from_key_states_module, 'bias') and fused_from_key_states_module.bias is not None:
            fused_from_key_states_module.bias.div_(k_scale.view(-1))
            get_logger().debug(f"bias of module {full_name}.{fused_unit.fused_from_key_states_name} divided")

        fused_from_query_states_module = getattr(submodule, fused_unit.fused_from_query_states_name)
        fused_from_query_states_module.weight.mul_(q_scale.view(-1, 1))
        get_logger().debug(f"weight of module {full_name}.{fused_unit.fused_from_query_states_name} multiplied")
        if hasattr(fused_from_query_states_module, 'bias') and fused_from_query_states_module.bias is not None:
            fused_from_query_states_module.bias.mul_(q_scale.view(-1))
            get_logger().debug(f"bias of module {full_name}.{fused_unit.fused_from_query_states_name} multiplied")

        get_logger().debug(f"Smoothing module fused to linear success: {full_name}")

    def _handle_module_fused_to_norm(self, full_name: str, submodule: nn.Module,
                                     fused_unit: KVSmoothFusedUnit, key_abs_max: torch.Tensor) -> None:
        get_logger().debug(f"Smoothing module fused to norm: {full_name}")
        self._check_module(full_name, submodule, fused_unit)

        head_dim = self.adapter.get_head_dim()
        num_key_value_heads = self.adapter.get_num_key_value_heads()
        get_logger().debug(
            f"head_dim: {head_dim}, "
            f"num_key_value_heads: {num_key_value_heads}, "
            f"key_abs_max.max: {key_abs_max.max()}, "
            f"key_abs_max.min: {key_abs_max.min()}, "
            f"key_abs_max.mean: {key_abs_max.mean()}")

        scale = key_abs_max.view(num_key_value_heads, 2, head_dim // 2).max(dim=1)[
            0]  # deal with rope: ____ ____ -> ____
        scale = scale.max(dim=0)[0].view(1, -1)  # norm has no head
        scale = scale.pow(self.config.smooth_factor)  # pow smooth factor
        scale = torch.repeat_interleave(scale, repeats=2, dim=0)  # recover rope: ____ -> ____ ____
        get_logger().debug(f"scale: {scale}")

        fused_from_key_states_module = getattr(submodule, fused_unit.fused_from_key_states_name)
        fused_from_key_states_module.weight.div_(scale.view(-1))
        get_logger().debug(f"weight of module {full_name}.{fused_unit.fused_from_key_states_name} divided")

        fused_from_query_states_module = getattr(submodule, fused_unit.fused_from_query_states_name)
        fused_from_query_states_module.weight.mul_(scale.view(-1))
        get_logger().debug(f"weight of module {full_name}.{fused_unit.fused_from_query_states_name} multiplied")

        get_logger().debug(f"Smoothing module fused to norm success: {full_name}")
