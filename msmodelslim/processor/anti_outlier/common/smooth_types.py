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
Smooth quantization types and configurations

This module contains data classes for smooth quantization algorithms including:
- IterSmooth: Iterative Smooth algorithm types
- FlexSmoothQuant: Flex Smooth Quantization algorithm types
- FlexAWQSSZ: Flex AWQ SSZ algorithm types
"""

from dataclasses import dataclass
from typing import Union, List, Optional, Dict, Any, TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from msmodelslim.core.quantizer.linear import LinearQConfig


@dataclass
class SmoothQuantContext:
    """SmoothQuant runtime context"""
    version: int
    a_smooth_scale: torch.Tensor
    shift: torch.Tensor


@dataclass
class SmoothQuantConfig:
    """SmoothQuant algorithm configuration"""
    version: int = 1
    alpha: float = 0.5
    shift: bool = False


@dataclass
class IterSmoothContext:
    version: int
    a_smooth_scale: torch.Tensor
    shift: torch.Tensor


@dataclass
class IterSmoothConfig:
    """

    iter_smooth算法的配置项。
    允许后续扩展配置项，但仅可新增新字段，且不得修改已有字段，
    version用于指定配置版本号，每次修改后，版本号需要加1。

    """

    version: int = 1
    alpha: float = 0.9
    shift: bool = False
    scale_min: float = 1e-5


@dataclass
class FlexSmoothQuantContext:
    version: int
    a_smooth_scale: torch.Tensor
    tensors: Optional[List[torch.Tensor]] = None


@dataclass
class FlexSmoothQuantConfig:
    """

    flex_smooth_quant算法的配置项。
    允许后续扩展配置项，但仅可新增新字段，且不得修改已有字段，
    version用于指定配置版本号，每次修改后，版本号需要加1。

    """

    version: int = 1
    alpha: Optional[float] = None
    beta: Optional[float] = None
    extra_config: Optional[Dict[str, Any]] = None


@dataclass
class FlexAWQSSZContext:
    version: int
    tensors: List[torch.Tensor]


@dataclass
class FlexAWQSSZConfig:
    """

    flex_awq_ssz算法的配置项。

    """

    version: int = 1
    alpha: Optional[float] = None
    beta: Optional[float] = None
    qconfig: Optional['LinearQConfig'] = None


SmoothContext = Union[SmoothQuantContext, IterSmoothContext, FlexSmoothQuantContext, FlexAWQSSZContext]

