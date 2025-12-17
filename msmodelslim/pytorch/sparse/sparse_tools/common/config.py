# Copyright Huawei Technologies Co., Ltd. 2023-2023. All rights reserved.
from dataclasses import dataclass
from typing import List


@dataclass
class Config:
    mode: str = 'sparse'
    
    def __post_init__(self):
        mode_list = ['sparse']
        if self.mode not in mode_list:
            raise ValueError(f"sparse method should be in {mode_list}")