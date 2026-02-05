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

"""Flex Smooth Quantization Processor (Refactored)"""

from typing import Callable, Any, Literal, Annotated, List, Optional, Dict

import torch.distributed as dist
import torch.nn as nn
from pydantic import AfterValidator, Field, model_validator
from pydantic_core import PydanticUndefined
from msmodelslim.ir.qal.qregistry import QABCRegistry
from msmodelslim.core.base.protocol import BatchProcessRequest
from msmodelslim.processor.base import AutoSessionProcessor, AutoProcessorConfig
from msmodelslim.core.quantizer.linear import LinearQConfig
from msmodelslim.core.observer import MsMinMaxObserver, MinMaxObserverConfig
from msmodelslim.utils.exception import UnsupportedError, SchemaValidateError
from msmodelslim.utils.logging import get_logger, logger_setter
from msmodelslim.utils.distributed.dist_ops import sync_gather_tensor_lists
from msmodelslim.utils.distributed import DistHelper
from msmodelslim.utils.validation.value import validate_normalized_value, is_string_list
from msmodelslim.ir.non_fusion_smooth_quant_ir import NonFusionSmoothQuantHookIR
from msmodelslim.ir.qal.qtypes import NonFusionSubgraph
from ..common import (
    FlexSmoothQuantConfig,
    FlexAWQSSZConfig,
    FlexSmoothQuantContext,
    FlexAWQSSZContext,
    StatsCollector,
    SubgraphRegistry,
    StatKey
)
from ..smooth_base import BaseSmoothProcessor
from .api import flex_smooth_quant, flex_awq_ssz
from .interface import FlexSmoothQuantInterface


class FlexSmoothBaseProcessorConfig(AutoProcessorConfig):
    """Base configuration class for Flex processors, defining common fields and validation rules"""
    type: Literal["_abstract_flex_smooth_base_"] = "_abstract_flex_smooth_base_"
    
    alpha: Annotated[float, AfterValidator(validate_normalized_value)] = None
    beta: Annotated[float, AfterValidator(validate_normalized_value)] = None
    enable_subgraph_type: Annotated[list, AfterValidator(is_string_list)] = Field(
        default_factory=lambda: ["norm-linear", "linear-linear", "ov", "up-down"]
    )
    include: Optional[List[str]] = None
    exclude: Optional[List[str]] = None


class FlexSmoothQuantProcessorConfig(FlexSmoothBaseProcessorConfig):
    """FlexSmoothQuant processor configuration"""
    type: Literal["flex_smooth_quant"] = "flex_smooth_quant"


class FlexAWQSSZProcessorConfig(FlexSmoothBaseProcessorConfig):
    """FlexAWQSSZ processor configuration"""
    qconfig: LinearQConfig = Field(description="量化配置")
    type: Literal["flex_awq_ssz"] = "flex_awq_ssz"

    @model_validator(mode="before")
    @classmethod
    def check_qconfig_missing(cls, values: dict) -> dict:
        """模型级前置校验：拦截 qconfig 缺失的场景"""
        if "qconfig" not in values or values["qconfig"] is PydanticUndefined:
            raise SchemaValidateError(
                "qconfig is a required parameter for flex_awq_ssz processor",
                action=(
                    "Please provide qconfig parameter in the YAML configuration, "
                    "including act and weight quantization settings"
                )
            )
        return values


class FlexStatsCollector(StatsCollector):
    """
    Flex smooth statistics collector
    """
    
    def __init__(self):
        super().__init__()
        self.dist_helper: Optional[DistHelper] = None
        # 为每个模块名称创建observer，用于收集channel_max统计
        self.observers: Dict[str, MsMinMaxObserver] = {}
    
    def set_dist_helper(self, dist_helper: Optional[DistHelper]):
        """设置分布式辅助类"""
        self.dist_helper = dist_helper
    
    def create_hook(self, name: str, subgraph_type: str = None) -> Callable:
        def stats_hook(module: nn.Linear, input_tensor: tuple, output: Any) -> None:
            # 有的路由专家可能采集不到激活，需要跳过
            if not input_tensor or not isinstance(input_tensor, tuple):
                get_logger().warning(f"Input tensor is empty for module {name}")
                return

            tensor = input_tensor[0]
            hidden_dim = tensor.shape[-1]
            tensor = tensor.reshape(-1, hidden_dim).detach()

            if name not in self.act_stats:
                self.act_stats[name] = {}
            statis_dict = self.act_stats[name]

            # 收集tensor用于后续算法
            cpu_tensor = tensor.to("cpu").reshape(-1, tensor.shape[-1])
            if StatKey.TENSOR not in statis_dict:
                statis_dict[StatKey.TENSOR] = [cpu_tensor]
            else:
                statis_dict[StatKey.TENSOR].append(cpu_tensor)

            if name not in self.observers:
                observer_config = MinMaxObserverConfig(dim=0, keepdim=False)
                self.observers[name] = MsMinMaxObserver(observer_config)
            # 根据模块是否共享决定是否同步
            sync = self.dist_helper.is_shared(name) if self.dist_helper is not None else False
            abs_tensor = tensor.abs()
            self.observers[name].update(abs_tensor, sync=sync)
            _, channel_max = self.observers[name].get_min_max()
            statis_dict[StatKey.STAT_KEY_SMOOTH_SCALE] = channel_max

        return stats_hook
    
    def clear_stats(self) -> None:
        super().clear_stats()
        for observer in self.observers.values():
            observer.reset()
        self.observers.clear()


class FlexSmoothBaseProcessor(BaseSmoothProcessor):    
    def __init__(self, model: nn.Module, config: FlexSmoothBaseProcessorConfig, adapter: object, **kwargs):
        super().__init__(model, config, adapter)
        self.config = config
        self._validate_parameters()
        self.stats_collector = FlexStatsCollector()
        
        # 初始化分布式辅助类（延迟到preprocess时创建，因为需要prefix信息）
        self.dist_helper = None

    def _validate_adapter_interface(self, adapter: object):
        """Validate that the adapter implements FlexSmoothQuantInterface."""
        if not isinstance(adapter, FlexSmoothQuantInterface):
            raise UnsupportedError(
                f'{adapter.__class__.__name__} does not implement FlexSmoothQuantInterface',
                action=f'Please ensure {adapter.__class__.__name__} inherits from FlexSmoothQuantInterface '
                       f'and implements the methods defined by the interface'
            )


@QABCRegistry.register(dispatch_key=FlexSmoothQuantProcessorConfig, abc_class=AutoSessionProcessor)
@logger_setter(prefix="msmodelslim.processor.flex_smooth_quant")
class FlexSmoothQuantProcessor(FlexSmoothBaseProcessor):
    """FlexSmoothQuant Processor"""

    def support_distributed(self) -> bool:
        return True
    
    def preprocess(self, request: BatchProcessRequest) -> None:
        # 在preprocess时创建DistHelper，传入prefix信息
        if dist.is_initialized():
            self.dist_helper = DistHelper(request.module, prefix=request.name)
            self.stats_collector.set_dist_helper(self.dist_helper)
        
        super().preprocess(request)
    
    def postprocess(self, request: BatchProcessRequest) -> None:
        super().postprocess(request)
        # 清理分布式辅助类
        self.stats_collector.set_dist_helper(None)
        self.dist_helper = None
    
    def apply_smooth_algorithm(self, subgraph_obj: Any, linear_names: List[str]) -> None:
        """Apply FlexSmoothQuant algorithm"""
        subgraph_type = SubgraphRegistry.get_name(type(subgraph_obj))
        config = FlexSmoothQuantConfig(
            alpha=self.config.alpha,
            beta=self.config.beta,
            extra_config=getattr(subgraph_obj, 'extra_config', None)
        )
        smooth_context = self._build_smooth_context(linear_names)
        if smooth_context is None:
            get_logger().warning(
                "No statistics collected for %s subgraph, skipping. "
                "This may happen for unused MOE experts.",
                subgraph_type
            )
            return
        scales = flex_smooth_quant(subgraph_obj, config, smooth_context)
        if scales is not None and isinstance(subgraph_obj, NonFusionSubgraph):
            for linear_module in subgraph_obj.linears:
                hook_ir = NonFusionSmoothQuantHookIR(scales)
                hook_handle = linear_module.register_forward_pre_hook(hook_ir)
                hook_ir.set_hook_handle(hook_handle)
        get_logger().info(f"Successfully applied FlexSmoothQuant to {subgraph_type} subgraph")

    def _build_smooth_context(self, linear_names: List[str]) -> Optional[FlexSmoothQuantContext]:
        a_smooth_scale = None
        tensors = None
        if not linear_names:
            get_logger().warning(
                "No linear modules provided while building FlexSmoothQuantContext; skipping smooth application."
            )
            return None
        # 仅用第一个linear的激活统计信息
        linear_name = linear_names[0]

        # 获取激活统计信息
        if linear_name in self.stats_collector.act_stats:
            stats = self.stats_collector.act_stats[linear_name]

            # 获取 smooth_scale
            if StatKey.STAT_KEY_SMOOTH_SCALE in stats:
                a_smooth_scale = stats[StatKey.STAT_KEY_SMOOTH_SCALE]
            else:
                a_smooth_scale = None

            # 获取 tensors
            if StatKey.TENSOR in stats:
                local_tensors = stats[StatKey.TENSOR]
            else:
                local_tensors = None
        else:
            get_logger().warning(f"Linear name {linear_name} not in act_stats")
            return None

        # 检查是否成功获取到激活平滑尺度
        if a_smooth_scale is None or local_tensors is None:
            # 返回 None 而不是抛出异常，让调用者决定如何处理
            get_logger().debug(
                "Failed to get activation smooth scale from linear name %s. "
                "This may happen for unused subgraphs (e.g., unactivated MOE experts).",
                linear_name
            )
            return None
        
        # 多卡同步：收集所有卡的 tensor，确保 alpha/beta 搜索使用相同的数据
        # 只有共享模块才需要进行多卡同步
        if (local_tensors and self.dist_helper is not None and self.dist_helper.is_shared(linear_name)):
            tensors = sync_gather_tensor_lists(local_tensors)
        else:
            # 单卡场景、非共享模块或空列表，直接使用本地 tensor
            tensors = local_tensors
        
        # 创建 FlexSmoothQuantContext
        smooth_context = FlexSmoothQuantContext(
            version=1,
            a_smooth_scale=a_smooth_scale,
            tensors=tensors,
        )

        return smooth_context


@QABCRegistry.register(dispatch_key=FlexAWQSSZProcessorConfig, abc_class=AutoSessionProcessor)
@logger_setter(prefix="msmodelslim.processor.flex_awq_ssz")
class FlexAWQSSZProcessor(FlexSmoothBaseProcessor):
    """FlexAWQSSZ Processor"""
    
    def apply_smooth_algorithm(self, subgraph_obj: Any, linear_names: List[str]) -> None:
        """Apply FlexAWQSSZ algorithm"""
        subgraph_type = SubgraphRegistry.get_name(type(subgraph_obj))
        config = FlexAWQSSZConfig(
            alpha=self.config.alpha,
            beta=self.config.beta,
            qconfig=self.config.qconfig
        )
        smooth_context = self._build_smooth_context(linear_names)
        if smooth_context is None:
            get_logger().warning(
                "No statistics collected for %s subgraph, skipping. "
                "This may happen for unused MOE experts.",
                subgraph_type
            )
            return
        flex_awq_ssz(subgraph_obj, config, smooth_context)
        get_logger().info(f"Successfully applied FlexAWQSSZ to {subgraph_type} subgraph")

    def _build_smooth_context(self, linear_names: List[str]) -> Optional[FlexAWQSSZContext]:
        tensors = None
        if not linear_names:
            get_logger().warning(
                "No linear modules provided while building FlexSmoothQuantContext; skipping smooth application."
            )
            return None
        # 仅用第一个linear的激活统计信息
        linear_name = linear_names[0]

        # 获取激活统计信息
        if linear_name in self.stats_collector.act_stats:
            stats = self.stats_collector.act_stats[linear_name]

            # 获取 tensors
            if StatKey.TENSOR in stats:
                tensors = stats[StatKey.TENSOR]
            else:
                tensors = None
        else:
            get_logger().warning(f"Linear name {linear_name} not in act_stats")
            return None

        # 检查是否成功获取到激活平滑尺度
        if tensors is None:
            # 返回 None 而不是抛出异常，让调用者决定如何处理
            get_logger().debug(
                "Failed to get activation tensors from linear name %s. "
                "This may happen for unused subgraphs (e.g., unactivated MOE experts).",
                linear_name
            )
            return None
        # 创建 FlexAWQSSZContext
        smooth_context = FlexAWQSSZContext(
            version=1,
            tensors=tensors
        )

        return smooth_context
