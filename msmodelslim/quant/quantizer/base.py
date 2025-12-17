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

from abc import abstractmethod

import torch
from pydantic import BaseModel, ConfigDict
from pydantic import validate_call
from torch import nn
from typing_extensions import Self, Optional, Dict, Any, Tuple

from msmodelslim.core.QAL.qbase import QStorage, QParam, QScheme, QScope, QDType
from msmodelslim.core.QAL.qregistry import QABCRegistry


class QConfig(BaseModel):
    dtype: QDType
    scope: QScope
    symmetric: bool
    method: str
    ext: Dict[str, Any] = {}

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

    def enable_sync(self):
        """
        启用同步操作
        子类可以重写此方法以实现更复杂的同步逻辑
        """
        self.sync = True
