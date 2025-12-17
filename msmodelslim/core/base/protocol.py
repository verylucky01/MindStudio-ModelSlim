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

from dataclasses import dataclass
from typing import Any, List, Dict, Tuple, Optional

from torch import nn


@dataclass
class DataUnit:
    input: Any
    output: Any


@dataclass
class ProcessRequest:
    """
    处理请求数据类，封装了处理器需要处理的事件信息。
    
    该类包含了处理事件、目标模块以及相关的数据，用于向处理器传递处理请求。
    
    属性:
        name: 目标模块的名称
        module: 目标模块，指定需要处理的PyTorch模块
        args: 处理数据列表，包含与处理事件相关的数据
        kwargs: 处理数据列表，包含与处理事件相关的数据
    """
    name: str
    module: nn.Module
    args: Tuple[Any]
    kwargs: Dict[str, Any]


@dataclass
class BatchProcessRequest:
    """
    批量处理请求数据类，封装了处理器需要处理的事件信息。
    
    该类包含了处理事件、目标模块以及相关的数据，用于向处理器传递处理请求。
    """
    name: str
    module: nn.Module
    datas: Optional[List[Tuple[Tuple[Any], Dict[str, Any]]]] = None
    outputs: Optional[List[Any]] = None
