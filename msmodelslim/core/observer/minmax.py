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


from typing import Optional, Tuple, Union, List

import torch
from pydantic import BaseModel
from torch import distributed as dist, nn

from msmodelslim.utils.distributed import sync_base_operation
from msmodelslim.utils.exception import SpecError


class MinMaxObserverConfig(BaseModel):
    dim: Union[int, List[int]] = []
    keepdim: bool = False


class MsMinMaxObserver(nn.Module):

    def __init__(self, config: MinMaxObserverConfig):
        super().__init__()
        self.config = config
        self.min_val = None
        self.max_val = None

    def update(self, x: torch.Tensor, sync: bool = False, group: Optional[dist.ProcessGroup] = None):

        if self.min_val is None:
            self.min_val = torch.amin(x, self.config.dim, self.config.keepdim)
        else:
            self.min_val = torch.min(self.min_val, torch.amin(x, self.config.dim, self.config.keepdim))

        if self.max_val is None:
            self.max_val = torch.amax(x, self.config.dim, self.config.keepdim)
        else:
            self.max_val = torch.max(self.max_val, torch.amax(x, self.config.dim, self.config.keepdim))

        if sync and dist.is_initialized():
            sync_base_operation(self.min_val, op='min')
            sync_base_operation(self.max_val, op='max')

    def reset(self):
        self.min_val = None
        self.max_val = None

    def get_min_max(self) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.min_val is None or self.max_val is None:
            raise SpecError(
                "Trying to get stats but no any update_stats invoked,"
                "maybe you are quantifying a moe expert, but this expert has never been activated.",
                action="Please check your model and quant config.")
        return self.min_val, self.max_val


class MinMaxBlockObserverConfig(BaseModel):
    method: str = 'max'
    clip: float = 1.0


class MsMinMaxBlockObserver(nn.Module):

    def __init__(self, config: MinMaxBlockObserverConfig):
        super().__init__()
        self.config = config
        self.min_val = None
        self.max_val = None

    def update(
            self,
            x: torch.Tensor,
            sync: bool = True,
            group: Optional[dist.ProcessGroup] = None,
            shared_exp_axes=None # 用于指定需要共享指数（或缩放因子）的维度
    ):
        if self.config.method == "max":
            if shared_exp_axes is None:
                # 若未指定共享维度，则直接计算输入张量x的全局绝对值最大值
                self.max_val = torch.max(torch.abs(x))
            else:
                # 若指定了共享维度，需沿这些维度聚合计算最大值（用于生成共享的指数/缩放因子）
                self.max_val = x
                for axis in shared_exp_axes:
                    # 需沿当前维度取绝对值的最大值，keepdim=True保持维度结构
                    # 该操作会将指定维度上的所有元素聚合为一个最大值，实现跨该维度的统计量共享
                    self.max_val, _ = torch.max(torch.abs(self.max_val), dim=axis, keepdim=True)
                    # 乘以配置的裁剪系数，对最大值进行限制（避免异常值影响）
                    self.max_val = self.max_val * self.config.clip
        elif self.config.method == 'none':
            self.max_val = torch.abs(x) # 若方法为'none'，则直接记录每个元素的绝对值作为max_val
        self.min_val = self.max_val.clone() # min_val暂用max_val的副本
        
        if sync and dist.is_initialized():
            sync_base_operation(self.min_val, op='min')
            sync_base_operation(self.max_val, op='max')

    def reset(self):
        self.min_val = None
        self.max_val = None

    def get_min_max(self) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.min_val is None or self.max_val is None:
            raise SpecError(
                "Trying to get stats but no any update_stats invoked,"
                "maybe you are quantifying a moe expert, but this expert has never been activated.",
                action="Please check your model and quant config.")
        return self.min_val, self.max_val
