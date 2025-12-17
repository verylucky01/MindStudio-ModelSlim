# Copyright Huawei Technologies Co., Ltd. 2023-2023. All rights reserved.
from dataclasses import dataclass
from msmodelslim.pytorch.sparse.sparse_tools.common.config import Config


@dataclass
class SparseConfig(Config):
    """ The configuration for Sparse."""
    mode: str = 'sparse'
    method: str = 'magnitude'
    sparse_ratio: float = 0.5
    progressive: bool = False
    uniform: bool = True
    
    def __post_init__(self):
        mode_list = ['sparse']
        if self.mode not in mode_list:
            raise ValueError("mode should be 'sparse'")
        
        method_list = ['magnitude', 'hessian', 'par', 'par_v2']
        if self.method not in method_list:
            raise ValueError(f"optimizer should be in {method_list}")
        if not isinstance(self.sparse_ratio, float):
            raise TypeError("sparse_ratio is invalid, please check it.")
        if (self.sparse_ratio <= 0) or (self.sparse_ratio >= 1):
            raise ValueError(f"sparse_ratio should be between 0 and 1")
        if not isinstance(self.progressive, bool):
            raise TypeError("progressive is invalid, please check it.")
        if not isinstance(self.uniform, bool):
            raise TypeError("uniform is invalid, please check it.")
