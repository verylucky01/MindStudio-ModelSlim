# -*- coding: utf-8 -*-
# Copyright Huawei Technologies Co., Ltd. 2023-2023. All rights reserved.
from abc import ABC, abstractmethod

from torch import nn as nn


class CompressModelWrapper(ABC):
    def __init__(self,
                 model: nn.Module,
                 cfg=None,
                 logger=None):
        super().__init__()
        self.model = model
        self.cfg = cfg
        self.logger = logger

    @staticmethod
    def _setattr(model: nn.Module,
                 module_name: str,
                 module: nn.Module):
        pass

    @abstractmethod
    def wrap(self):
        raise NotImplementedError
