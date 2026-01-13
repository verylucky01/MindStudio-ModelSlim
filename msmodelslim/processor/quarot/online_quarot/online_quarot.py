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

from typing import List, Literal, Optional, Dict, Callable

import torch
import torch.nn as nn
from pydantic import field_validator

import msmodelslim.ir as qir
from msmodelslim.ir.qal.qregistry import QABCRegistry
from msmodelslim.core.base.protocol import BatchProcessRequest
from msmodelslim.processor.base import AutoProcessorConfig, AutoSessionProcessor
from msmodelslim.utils.config_map import ConfigSet
from msmodelslim.utils.exception import SchemaValidateError, UnsupportedError
from msmodelslim.utils.logging import get_logger
from msmodelslim.ir.quarot import RotationType
from .online_quarot_interface import OnlineQuaRotInterface, RotationConfig
from ..common.quarot_utils import QuaRotMode, create_rot, is_power_of_two, rotate_linear


class OnlineQuaRotProcessorConfig(AutoProcessorConfig):
    """简化的在线旋转处理器配置类"""
    type: Literal["online_quarot"] = "online_quarot"
    include: Optional[List[str]] = None  # 包含模式列表
    exclude: Optional[List[str]] = None  # 排除模式列表
    block_size: int = -1  # 块大小（默认值，可被 RotationConfig 中的 block_size 覆盖）

    @field_validator('block_size')
    @classmethod
    def validate_block_size(cls, v):
        """校验 block_size：取值范围为-1或大于0且为2的幂的整数"""
        if v == -1:
            return v
        if v <= 0 or not is_power_of_two(v):
            raise SchemaValidateError(
                f"block_size must be -1 or a positive power of 2, got {v}"
            )
        return v


def _get_full_module_name(target_module: nn.Module, request: BatchProcessRequest) -> str:
    """
    通过遍历request.module获取完整的模块名称。
    """
    for name, module in request.module.named_modules():
        if module is target_module:
            # 拼接完整路径：request.name + 相对路径
            full_name = f"{request.name}.{name}" if name else request.name
            return full_name

    # 如果找不到，抛出UnsupportedError
    raise UnsupportedError(f"Cannot find full module name for {target_module}")


def _convert_hookir_to_wrapper(model: nn.Module) -> None:
    """
    将模型中的HookIR转换为Wrapper

    Args:
        model: 要处理的模型
    """
    # 遍历模型中的所有子模块
    for name, sub_module in model.named_modules():
        if hasattr(sub_module, '_forward_pre_hooks'):
            # 遍历模块的所有前向钩子
            for hook in sub_module._forward_pre_hooks.values():
                # 检查是否是HookIR类型
                if isinstance(hook, qir.HookIR):
                    # 将hook_ir转换为wrapper
                    wrapper = hook.wrapper_module(sub_module)
                    # 将wrapper替换模块
                    model.set_submodule(name, wrapper)
                    get_logger().info(f"Converted {type(hook)} to wrapper for module: {name}")
        
        if hasattr(sub_module, '_forward_hooks'):
            # 遍历模块的所有后向钩子
            for hook in sub_module._forward_hooks.values():
                # 检查是否是HookIR类型
                if isinstance(hook, qir.HookIR):
                    # 将hook_ir转换为wrapper
                    wrapper = hook.wrapper_module(sub_module)
                    # 将wrapper替换模块
                    model.set_submodule(name, wrapper)
                    get_logger().info(f"Converted {type(hook)} to wrapper for module: {name}")


@QABCRegistry.register(dispatch_key=OnlineQuaRotProcessorConfig, abc_class=AutoSessionProcessor)
class OnlineQuaRotProcessor(AutoSessionProcessor):
    """
    简化的在线旋转处理器。
    
    支持四种旋转模式：
    - input: 在模块输入处旋转（使用 forward_pre_hook）
    - output: 在模块输出处旋转（使用 forward_hook）
    - replace: 替换模块，只执行旋转（使用 WrapperIR）
    - offline: 离线旋转，直接作用在模型参数上（使用 rotate_linear）
    """

    def __init__(
            self,
            model: nn.Module,
            config: OnlineQuaRotProcessorConfig,
            adapter: OnlineQuaRotInterface,
            **kwargs
    ) -> None:
        super().__init__(model)
        self.config = config
        self.model = model
        self.adapter = adapter
        
        if not isinstance(adapter, OnlineQuaRotInterface):
            raise UnsupportedError(
                f'{adapter.__class__.__name__} does not support OnlineQuaRotInterface',
                action='Please provide a valid model adapter '
                       'which implements OnlineQuaRotInterface'
            )
        
        # 获取旋转配置（传递 model 以便 adapter 可以挂载模块）
        self.rotation_configs: Dict[str, RotationConfig] = adapter.get_online_rotation_configs(model)
        
        # 初始化 include/exclude 过滤
        self.include_set = ConfigSet(config.include) if config.include else ConfigSet(["*"])
        self.exclude_set = ConfigSet(config.exclude) if config.exclude else ConfigSet([])
        
        # 存储旋转信息
        self.rotation_infos: Dict[str, qir.OnlineRotationInfo] = {}
        
        # 初始化旋转策略映射
        self._rotation_strategies = self._get_rotation_strategies()

    def support_distributed(self) -> bool:
        return True

    def is_data_free(self) -> bool:
        return True

    def pre_run(self) -> None:
        """预处理阶段：初始化旋转矩阵"""
        get_logger().info("Initializing online rotation matrices")
        
        # 为每个配置生成或验证旋转矩阵
        for module_name, rot_config in self.rotation_configs.items():
            if not self._should_apply_rotation(module_name):
                continue
            
            # 获取或生成旋转矩阵
            if rot_config.rotation_matrix is not None:
                # 用户提供的旋转矩阵
                rotation_matrix = rot_config.rotation_matrix
            else:
                # 自动生成旋转矩阵
                if rot_config.rotation_mode is None:
                    raise UnsupportedError(
                        f"Rotation matrix for {module_name} is not provided and "
                        "rotation_mode is not specified in RotationConfig",
                        action="Please provide rotation_matrix or specify rotation_mode in RotationConfig"
                    )
                
                rotation_matrix = create_rot(
                    mode=rot_config.rotation_mode,
                    size=rot_config.rotation_size,
                    block_size=rot_config.block_size if rot_config.block_size != -1 else self.config.block_size,
                    rot_step=rot_config.rot_step,
                    eye_step=rot_config.eye_step,
                    seed=rot_config.seed,
                    dtype=rot_config.dtype,
                )
            
            # 将字符串类型的 rotation_type 转换为 RotationType 枚举
            rotation_type_enum = RotationType(rot_config.rotation_type)
            
            # 创建旋转信息对象
            rotation_info = qir.OnlineRotationInfo(
                rotation_matrix=rotation_matrix,
                rotation_type=rotation_type_enum,
                layer_name=module_name,
                rotation_side=rot_config.rotation_side
            )
            self.rotation_infos[module_name] = rotation_info
        
        get_logger().info(f"Initialized {len(self.rotation_infos)} rotation configurations")

    def preprocess(self, request: BatchProcessRequest) -> None:
        """为每个层应用旋转"""
        prefix = request.name
        prefix = f"{prefix}." if prefix != "" else ""
        
        # 过滤出当前层范围内的旋转配置
        filtered_configs = {}
        for module_name, rot_config in self.rotation_configs.items():
            if module_name.startswith(prefix):
                # 移除前缀，获取相对名称
                relative_name = module_name[len(prefix):] if prefix else module_name
                filtered_configs[relative_name] = (module_name, rot_config)
        
        # 为每个配置应用旋转
        for relative_name, (full_module_name, rot_config) in filtered_configs.items():
            if not self._should_apply_rotation(full_module_name):
                continue
            
            try:
                # 获取模块
                module = request.module.get_submodule(relative_name)
                rotation_info = self.rotation_infos[full_module_name]
                
                # 根据旋转类型应用旋转
                self._apply_rotation(module, full_module_name, rotation_info, request)
            except AttributeError:
                get_logger().warning(
                    f"Module {relative_name} not found in {request.name}, skipping rotation"
                )
            except Exception as e:
                get_logger().error(
                    f"Failed to apply rotation to {full_module_name}: {e}",
                    exc_info=True
                )

    def post_run(self) -> None:
        """后处理阶段：转换 HookIR 为 WrapperIR"""
        _convert_hookir_to_wrapper(self.model)
        get_logger().info("Converted all HookIR to WrapperIR")

    def _should_apply_rotation(self, module_name: str) -> bool:
        """
        检查是否应该对模块应用旋转。
        """
        # 检查 include
        if module_name not in self.include_set:
            return False
        
        # 检查 exclude（优先级高于 include）
        if module_name in self.exclude_set:
            return False
        
        return True

    def _get_rotation_strategies(self) -> Dict[RotationType, Callable]:
        """
        返回旋转类型到处理方法的映射
        """
        return {
            RotationType.INPUT: self._apply_input_rotation,
            RotationType.OUTPUT: self._apply_output_rotation,
            RotationType.REPLACE: self._apply_replace_rotation,
            RotationType.OFFLINE: self._apply_offline_rotation,
        }

    def _apply_input_rotation(
            self,
            module: nn.Module,
            full_module_name: str,
            rotation_info: qir.OnlineRotationInfo,
            request: BatchProcessRequest
    ) -> None:
        """
        处理 INPUT 类型旋转：在模块输入处旋转（使用 forward_pre_hook）。
        """
        hook_ir = qir.OnlineRotationInputHookIR(full_module_name, rotation_info)
        hook_handle = module.register_forward_pre_hook(hook_ir)
        hook_ir.set_hook_handle(hook_handle)
        get_logger().debug(f"Applied input rotation to {full_module_name}")

    def _apply_output_rotation(
            self,
            module: nn.Module,
            full_module_name: str,
            rotation_info: qir.OnlineRotationInfo,
            request: BatchProcessRequest
    ) -> None:
        """
        处理 OUTPUT 类型旋转：在模块输出处旋转（使用 forward_hook）。
        """
        hook_ir = qir.OnlineRotationOutputHookIR(full_module_name, rotation_info)
        hook_handle = module.register_forward_hook(hook_ir)
        hook_ir.set_hook_handle(hook_handle)
        get_logger().debug(f"Applied output rotation to {full_module_name}")

    def _apply_replace_rotation(
            self,
            module: nn.Module,
            full_module_name: str,
            rotation_info: qir.OnlineRotationInfo,
            request: BatchProcessRequest
    ) -> None:
        """
        处理 REPLACE 类型旋转：替换模块，只执行旋转（使用 WrapperIR）。
        """
        wrapper = qir.OnlineRotationWrapper(module, full_module_name, rotation_info)
        # 获取相对名称以替换模块
        relative_name = full_module_name[len(request.name) + 1:] if request.name and full_module_name.startswith(request.name + ".") else full_module_name
        # 使用 request.module 的 set_submodule 方法
        request.module.set_submodule(relative_name, wrapper)
        get_logger().debug(f"Applied replace rotation to {full_module_name}")

    def _apply_offline_rotation(
            self,
            module: nn.Module,
            full_module_name: str,
            rotation_info: qir.OnlineRotationInfo,
            request: BatchProcessRequest
    ) -> None:
        """
        处理 OFFLINE 类型旋转：直接作用在模型参数上（使用 rotate_linear）。
        """
        if not hasattr(module, 'weight') or module.weight is None or module.weight.dim() != 2:
            raise UnsupportedError(
                f"Offline rotation only supports modules with weight attribute and weight dimension is 2, "
                f"got {type(module)} for {full_module_name}",
                action=f"Please use offline rotation only on modules with weight attribute and weight dimension is 2"
            )
        
        # 获取旋转方向和矩阵
        rotation_side = rotation_info.rotation_side or "right"
        right_rotate = (rotation_side == "right")
        rotation_matrix = rotation_info.rotation_matrix
        
        # 应用旋转到线性层权重
        try:
            rotate_linear(module, rotation_matrix, right_rotate=right_rotate)
            get_logger().debug(
                f"Applied offline {rotation_side} rotation to {full_module_name}"
            )
        except UnsupportedError as e:
            raise UnsupportedError(
                f"Failed to apply offline rotation to {full_module_name}: {e}",
                action=f"Please check the rotation matrix size matches the weight dimensions"
            ) from e

    def _handle_unknown_rotation_type(
            self,
            rotation_type: RotationType,
            full_module_name: str
    ) -> None:
        raise UnsupportedError(
            f"Unknown rotation type: {rotation_type}",
            action=f"Please use one of ['input', 'output', 'replace', 'offline'] for {full_module_name}"
        )

    def _apply_rotation(
            self,
            module: nn.Module,
            full_module_name: str,
            rotation_info: qir.OnlineRotationInfo,
            request: BatchProcessRequest
    ) -> None:
        """
        应用旋转到模块。
        
        根据 rotation_type 分发到对应的策略方法。子类可以通过重写
        `_get_rotation_strategies()` 来扩展新的旋转类型，或重写特定的
        策略方法来修改处理逻辑。
        """
        rotation_type = rotation_info.rotation_type
        strategy = self._rotation_strategies.get(rotation_type)
        
        if strategy:
            strategy(module, full_module_name, rotation_info, request)
        else:
            self._handle_unknown_rotation_type(rotation_type, full_module_name)

