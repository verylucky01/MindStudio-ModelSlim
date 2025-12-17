#  -*- coding: utf-8 -*-
#  Copyright (c) 2024-2024 Huawei Technologies Co., Ltd.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#  http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.


from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any, Callable
from enum import Enum
import torch
import torch.nn as nn

from msmodelslim.utils.logging import get_logger
from msmodelslim.core.QAL.qtypes import (
    NormLinearSubgraph,
    LinearLinearSubgraph,
    OVSubgraph,
    UpDownSubgraph
)


class StatKey(str, Enum):
    STAT_KEY_MAX = "max"
    STAT_KEY_MIN = "min"
    STAT_KEY_SHIFT = "shift"
    STAT_KEY_THRESHOLD_CHANNEL = "thres_c"
    STAT_KEY_THRESHOLD_TENSOR = "thres_t"
    STAT_KEY_SMOOTH_SCALE_MASK = "smooth_scale_mask"
    STAT_KEY_SMOOTH_SCALE = "smooth_scale"
    STAT_KEY_VARIANCE = "std"
    TENSOR = 'tensor'


class HookManager:   
    def __init__(self, model: nn.Module):
        self.model = model
        self.hook_handles: Dict[str, any] = {}
    
    def install_hook(self, module_name: str, hook_fn: Callable, subgraph_type: str = None) -> bool:
        module = self.model.get_submodule(module_name)
        handle = module.register_forward_hook(hook_fn)
        self.hook_handles[module_name] = handle
        get_logger().debug(
            f"Successfully installed hook for module {module_name}"
            + (f" (subgraph_type: {subgraph_type})" if subgraph_type else "")
        )
        return True
    
    def remove_all_hooks(self) -> int:   
        for module_name, handle in self.hook_handles.items():
            handle.remove()
            get_logger().debug("Successfully removed hook for module %s", module_name)
        self.hook_handles.clear()
        return len(self.hook_handles)


class StatsCollector(ABC):
    def __init__(self):
        self.act_stats: Dict[str, Dict[str, Any]] = {}
    
    @abstractmethod
    def create_hook(self, name: str, subgraph_type: str = None) -> Callable:
        pass
    
    def clear_stats(self) -> None:
        self.act_stats.clear()


class SubgraphRegistry:
    SUBGRAPH_TO_NAME = {
        NormLinearSubgraph: "norm-linear",
        LinearLinearSubgraph: "linear-linear",
        OVSubgraph: "ov",
        UpDownSubgraph: "up-down"
    }

    NAME_TO_HANDLER = {
        "norm-linear": "_apply_norm_linear_smooth",
        "linear-linear": "_apply_linear_linear_smooth",
        "ov": "_apply_ov_smooth",
        "up-down": "_apply_up_down_smooth",
    }

    SUPPORTED_TYPES = ["norm-linear", "linear-linear", "ov", "up-down"]
    
    @classmethod
    def get_name(cls, subgraph_type: type) -> str:
        return cls.SUBGRAPH_TO_NAME.get(subgraph_type, "unknown")
    
    @classmethod
    def get_handler_name(cls, subgraph_name: str) -> Optional[str]:
        return cls.NAME_TO_HANDLER.get(subgraph_name)
    
    @classmethod
    def is_supported(cls, subgraph_name: str) -> bool:
        return subgraph_name in cls.SUPPORTED_TYPES
    
    @classmethod
    def get_all_supported_types(cls) -> list:
        return cls.SUPPORTED_TYPES.copy()
