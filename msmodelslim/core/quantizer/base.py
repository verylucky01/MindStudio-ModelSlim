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

from abc import abstractmethod

import torch
from pydantic import BaseModel, ConfigDict, Field
from pydantic import validate_call
from torch import nn
from typing_extensions import Self, Optional, Dict, Any, Tuple

from msmodelslim.ir.qal.qbase import QStorage, QParam, QScheme, QScope, QDType
from msmodelslim.ir.qal.qregistry import QABCRegistry


class QConfig(BaseModel):
    dtype: QDType
    scope: QScope
    symmetric: bool
    method: str
    ext: Dict[str, Any] = Field(default_factory=dict, exclude_if=lambda v: not v)

    model_config = ConfigDict(extra="forbid")

    def to_scheme(self):
        return QScheme(QScope(self.scope), QDType(self.dtype), self.symmetric)


@QABCRegistry.register_abc(dispatch_key=Tuple[QScheme, str])
class AutoActQuantizer(nn.Module):

    def __init__(self):
        super().__init__()
        self.sync = False  # 默认不启用同步操作

    @classmethod
    @validate_call(config=dict(arbitrary_types_allowed=True))
    def from_config(cls, config: QConfig) -> Self:
        return QABCRegistry.create(
            AutoActQuantizer,
            (config.to_scheme(), config.method),
            *(config,)
        )

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pass

    @abstractmethod
    def get_q_param(self) -> QParam:
        """
        获取量化参数
        """
        pass

    def support_distributed(self) -> bool:
        """
        判断是否支持分布式
        
        Returns:
            bool: 是否支持分布式，默认为True
        """
        return True

    def is_data_free(self) -> bool:
        """
        判断是否data free场景
        
        Returns:
            bool: 是否是否data free场景，默认为False
        """
        return False

    def enable_sync(self):
        """
        启用同步操作
        子类可以重写此方法以实现更复杂的同步逻辑
        """
        self.sync = True


@QABCRegistry.register_abc(dispatch_key=Tuple[QScheme, str])
class AutoWeightQuantizer(nn.Module):

    def __init__(self):
        super().__init__()
        self.sync = False  # 默认不启用同步操作

    @classmethod
    @validate_call(config=dict(arbitrary_types_allowed=True))
    def from_config(cls, config: QConfig) -> Self:
        return QABCRegistry.create(
            AutoWeightQuantizer,
            (config.to_scheme(), config.method),
            *(config,)
        )

    @abstractmethod
    def init_weight(self, weight: QStorage, bias: Optional[torch.Tensor] = None) -> None:
        """
        初始化权重Tensor

        Args:
            bias: 偏移
            weight: 权重

        Returns:
            None
        """

        pass

    @abstractmethod
    def forward(self, x: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        对权重进行量化和反量化

        Args:
            x: 激活值

        Returns:
            torch.Tensor: 量化然后反量化后的激活值，与init_weight所提供的权重shape/dtype相同
        """

        pass

    @abstractmethod
    def get_q_storage(self) -> QStorage:
        """
        获取量化后的权重
        """
        pass

    @abstractmethod
    def get_q_param(self) -> QParam:
        """
        获取量化参数
        """
        pass

    def support_distributed(self) -> bool:
        """
        判断是否支持分布式
        
        Returns:
            bool: 是否支持分布式，默认为True
        """
        return True

    def is_data_free(self) -> bool:
        """
        判断是否data free场景
        
        Returns:
            bool: 是否是否data free场景，默认为True
        """
        return True

    def enable_sync(self):
        """
        启用同步操作
        子类可以重写此方法以实现更复杂的同步逻辑
        """
        self.sync = True
