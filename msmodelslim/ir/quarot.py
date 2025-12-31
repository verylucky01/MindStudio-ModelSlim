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

from typing import Any, Dict, List, Optional, Tuple

import torch
from torch import nn

from msmodelslim.utils.logging import logger_setter
from .wrapper import WrapperIR, HookIR


class QuarotOnlineRotationInfo:
    """
    Quarot旋转矩阵信息。
    
    该类负责管理全局共享的旋转矩阵和层索引信息。
    """

    def __init__(
            self,
            rotation_o_proj: Optional[torch.Tensor],
            rotation_o_proj_eye: Optional[torch.Tensor],
            rotation_down_proj_m: Optional[torch.Tensor],
            rotation_down_proj_n: Optional[torch.Tensor],
            max_tp_size: int,
    ):
        """
        初始化QuarotRotationInfo。
        
        Args:
            rotation_o_proj: 普通旋转矩阵
            rotation_down_proj_m: Kronecker旋转矩阵M
            rotation_down_proj_n: Kronecker旋转矩阵N
        """
        self.heads_rotation = rotation_o_proj
        self.heads_rotation_eye = rotation_o_proj_eye
        self.kronecker_rotation_m = rotation_down_proj_m
        self.kronecker_rotation_n = rotation_down_proj_n
        self.max_tp_size = max_tp_size

        self.heads_rotation_layers: List[str] = []
        self.kronecker_rotation_layers: List[str] = []

    def add_rotation_layer(self, layer_name: str) -> None:
        """添加使用全局旋转矩阵的层名称。"""
        self.heads_rotation_layers.append(layer_name)

    def add_kronecker_rotation_layer(self, layer_name: str) -> None:
        """添加使用全局Kronecker旋转矩阵的层名称。"""
        self.kronecker_rotation_layers.append(layer_name)

    def get_quarot_save_info(self) -> Dict[str, Any]:
        """
        获取quarot相关的保存信息。
        
        Returns:
            包含旋转矩阵和层名称的字典
        """
        return {
            "max_tp_size": self.max_tp_size,
            "heads_rotation": {
                "layers": self.heads_rotation_layers.copy()
            },
            "kronecker_rotation": {
                "layers": self.kronecker_rotation_layers.copy()
            }
        }


@logger_setter()
class QuarotOnlineHeadRotationWrapper(WrapperIR):
    """
    直接进行旋转运算的包装器。
    
    该类继承自WrapperIR，包装AutoFakeQuantLinear实例，使用全局共享的旋转矩阵，
    在forward前添加旋转运算。
    """

    def __init__(
            self,
            module: nn.Module,
            layer_name: str,
            rotation_info: QuarotOnlineRotationInfo
    ):
        """
        初始化RotationWrapper包装器。
        
        Args:
            module: 被包装的AutoFakeQuantLinear实例
            layer_name: 层名称，用于保存时标识
            rotation_info: 旋转矩阵信息
        """
        super().__init__(module)
        self.layer_name = layer_name
        self.rotation_info = rotation_info
        self.rotation_info.add_rotation_layer(layer_name)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播，在AutoFakeQuantLinear前添加旋转运算。
        
        Args:
            x: 输入张量
            
        Returns:
            经过旋转运算和线性变换的输出张量
        """
        x_rotated = self._apply_rotation(x)
        return self.wrapped_module(x_rotated)

    def extra_repr(self) -> str:
        """
        返回额外的字符串表示，描述旋转矩阵信息。

        Returns:
            包含旋转矩阵信息的字符串
        """
        rot_1 = self.rotation_info.heads_rotation
        rot_2 = self.rotation_info.heads_rotation_eye

        return f"heads_rotation(Q:{rot_1.shape[0]}x{rot_1.shape[1]}, I:{rot_2.shape[0]}x{rot_2.shape[1]})"

    def _apply_rotation(self, x: torch.Tensor) -> torch.Tensor:
        """
        应用旋转运算。
        
        Args:
            x: 输入张量
            
        Returns:
            旋转后的张量
        """
        rot_1 = self.rotation_info.heads_rotation
        rot_2 = self.rotation_info.heads_rotation_eye
        dtype = x.dtype
        device = x.device
        init_shape = x.shape
        scaled_x = x.reshape(-1, rot_1.shape[0], rot_2.shape[0])
        scaled_x = torch.matmul(rot_1.to(device, dtype).T, scaled_x).reshape(init_shape)
        return scaled_x.reshape(init_shape)


@logger_setter()
class QuarotOnlineKroneckerRotationWrapper(WrapperIR):
    """
    按Kronecker Product方式进行旋转的包装器。
    
    该类继承自WrapperIR，包装AutoFakeQuantLinear实例，使用全局共享的旋转矩阵，
    通过Kronecker Product组合后进行旋转运算。
    """

    def __init__(
            self,
            module: nn.Module,
            layer_name: str,
            rotation_info: QuarotOnlineRotationInfo
    ):
        """
        初始化KroneckerRotationWrapper包装器。
        
        Args:
            module: 被包装的AutoFakeQuantLinear实例
            layer_name: 层名称，用于保存时标识
            rotation_info: 旋转矩阵信息
        """
        super().__init__(module)
        self.layer_name = layer_name
        self.rotation_info = rotation_info
        self.rotation_info.add_kronecker_rotation_layer(layer_name)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播，在AutoFakeQuantLinear前添加Kronecker旋转运算。
        
        Args:
            x: 输入张量
            
        Returns:
            经过Kronecker旋转运算和线性变换的输出张量
        """
        x_rotated = self._apply_kronecker_rotation(x)
        return self.wrapped_module(x_rotated)

    def extra_repr(self) -> str:
        """
        返回额外的字符串表示，描述Kronecker旋转矩阵信息。

        Returns:
            包含Kronecker旋转矩阵信息的字符串
        """
        rot_1 = self.rotation_info.kronecker_rotation_m
        rot_2 = self.rotation_info.kronecker_rotation_n

        return f"kronecker_rotation(M:{rot_1.shape[0]}x{rot_1.shape[1]}, N:{rot_2.shape[0]}x{rot_2.shape[1]})"

    def _apply_kronecker_rotation(self, x: torch.Tensor) -> torch.Tensor:
        """
        应用Kronecker Product旋转运算。
        
        Args:
            x: 输入张量
            
        Returns:
            Kronecker旋转后的张量
        """
        rot_1 = self.rotation_info.kronecker_rotation_m
        rot_2 = self.rotation_info.kronecker_rotation_n
        dtype = x.dtype
        device = x.device
        init_shape = x.shape
        scaled_x = x.reshape(-1, rot_1.shape[0], rot_2.shape[0])
        scaled_x = torch.matmul(scaled_x, rot_2.to(device, dtype))
        scaled_x = torch.matmul(rot_1.to(device, dtype).T, scaled_x).reshape(init_shape)
        return scaled_x.reshape(init_shape)


@logger_setter()
class QuarotHeadsRotationHookIR(HookIR):
    """
    Quarot专用的HookIR实现，用于替代直接使用register_forward_pre_hook。
    
    该类实现了HookIR抽象基类，将hook信息转换为QuarotOnlineRotationWrapper。
    """

    def __init__(self, layer_name: str, rotation_info: QuarotOnlineRotationInfo):
        """
        初始化QuarotHookIR。
        
        Args:
            layer_name: 层名称
            rotation_info: 旋转矩阵信息
        """
        super().__init__()
        self.layer_name = layer_name
        self.rotation_info = rotation_info

    def __call__(
            self,
            module: nn.Module,
            args: Tuple[Any, ...],
    ) -> Tuple[Any, ...]:
        """
        实现Callable接口，作为hook函数被调用。
        
        Args:
            module: 被hook的模块
            args: 模块的输入参数

        Returns:
            处理后的输入参数和关键字参数
        """
        # 执行普通旋转运算
        dtype = args[0].dtype
        device = args[0].device
        x = args[0]
        init_shape = x.shape

        # 应用旋转运算
        rot_1 = self.rotation_info.heads_rotation
        rot_2 = self.rotation_info.heads_rotation_eye

        scaled_x = x.reshape(-1, rot_1.shape[0], rot_2.shape[0])
        scaled_x = torch.matmul(rot_1.to(device, dtype).T, scaled_x).reshape(init_shape)

        return (scaled_x,) + args[1:]

    def wrapper_module(self, module: nn.Module) -> WrapperIR:
        """
        实现HookIR抽象方法，返回QuarotOnlineRotationWrapper。
        
        Args:
            module: 要包装的模块
            
        Returns:
            QuarotOnlineRotationWrapper实例
        """
        # 将hook信息转换为WrapperIR
        self.remove_hook()
        return QuarotOnlineHeadRotationWrapper(module, self.layer_name, self.rotation_info)


@logger_setter()
class QuarotKroneckerRotationHookIR(HookIR):
    """
    Quarot专用的Kronecker HookIR实现，用于down_proj的Kronecker旋转。
    
    该类实现了HookIR抽象基类，将hook信息转换为QuarotOnlineKroneckerRotationWrapper。
    """

    def __init__(self, layer_name: str, rotation_info: QuarotOnlineRotationInfo):
        """
        初始化QuarotKroneckerHookIR。
        
        Args:
            layer_name: 层名称
            rotation_info: 旋转矩阵信息
        """
        super().__init__()
        self.layer_name = layer_name
        self.rotation_info = rotation_info

    def __call__(
            self,
            module: nn.Module,
            args: Tuple[Any, ...],
    ) -> Tuple[Any, ...]:
        """
        实现Callable接口，作为hook函数被调用。
        
        Args:
            module: 被hook的模块
            args: 模块的输入
            
        Returns:
            处理后的输入
        """
        # 执行Kronecker旋转运算
        dtype = args[0].dtype
        device = args[0].device
        x = args[0]
        init_shape = x.shape

        # 使用rotation_info中的Kronecker旋转矩阵进行运算
        rot_1 = self.rotation_info.kronecker_rotation_m
        rot_2 = self.rotation_info.kronecker_rotation_n

        scaled_x = x.reshape(-1, rot_1.shape[0], rot_2.shape[0])
        scaled_x = torch.matmul(scaled_x, rot_2.to(device, dtype))
        scaled_x = torch.matmul(rot_1.to(device, dtype).T, scaled_x).reshape(init_shape)

        return (scaled_x,) + args[1:]

    def wrapper_module(self, module: nn.Module) -> WrapperIR:
        """
        实现HookIR抽象方法，返回QuarotOnlineKroneckerRotationWrapper。
        
        Args:
            module: 要包装的模块
            
        Returns:
            QuarotOnlineKroneckerRotationWrapper实例
        """
        # 将hook信息转换为WrapperIR
        self.remove_hook()
        return QuarotOnlineKroneckerRotationWrapper(module, self.layer_name, self.rotation_info)
