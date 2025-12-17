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
from abc import ABC, abstractmethod
from enum import Enum
from typing import List

from pydantic import BaseModel

from msmodelslim.utils.exception import UnsupportedError


class KVSmoothFusedType(Enum):
    StateViaRopeToNorm = 'state-rope-norm'
    StateViaRopeToLinear = 'state-rope-linear'


class KVSmoothFusedUnit(BaseModel):
    attention_name: str  # specific whole name of attention layer, e.g. "model.layers.0.self_attn"
    layer_idx: int  # specific layer index, e.g. 0
    fused_from_query_states_name: str  # specific base name of module fused from query_states, e.g. "q_proj"
    fused_from_key_states_name: str  # specific base name of module fused from key_states, e.g. "k_proj"
    fused_type: KVSmoothFusedType  # fused type, support state-rope-norm and state-rope-linear for now


class KVSmoothFusedInterface(ABC):
    @abstractmethod
    def get_kvcache_smooth_fused_subgraph(self) -> List[KVSmoothFusedUnit]:
        raise UnsupportedError(f'{self.__class__.__name__} does not support get_kvcache_smooth_fused_subgraph',
                               action='Please implement this method before using kvcache smooth')

    @abstractmethod
    def get_head_dim(self) -> int:
        raise UnsupportedError(f'{self.__class__.__name__} does not support get_head_dim',
                               action='Please implement this method before using kvcache smooth')

    @abstractmethod
    def get_num_key_value_groups(self) -> int:
        raise UnsupportedError(f'{self.__class__.__name__} does not support get_num_key_value_groups',
                               action='Please implement this method before using kvcache smooth')

    @abstractmethod
    def get_num_key_value_heads(self) -> int:
        raise UnsupportedError(f'{self.__class__.__name__} does not support get_num_key_value_heads',
                               action='Please implement this method before using kvcache smooth')
