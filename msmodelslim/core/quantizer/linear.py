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

from typing import Optional

import torch
import torch.nn.functional as F
from pydantic import BaseModel, ConfigDict
from pydantic import validate_call
from torch import nn

import msmodelslim.ir as qir
from msmodelslim.ir.qal import QDType, QStorage
from msmodelslim.utils.logging import logger_setter
from .base import AutoActQuantizer, AutoWeightQuantizer, QConfig


class LinearQConfig(BaseModel):
    act: QConfig
    weight: QConfig

    model_config = ConfigDict(extra="forbid")


@logger_setter()
class LinearQuantizer(nn.Module):

    @validate_call(config=dict(arbitrary_types_allowed=True))
    def __init__(self, config: LinearQConfig, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.config = config
        self.sync = False  # 默认不启用同步操作
        self.input_quantizer = AutoActQuantizer.from_config(config.act)
        self.weight_quantizer = AutoWeightQuantizer.from_config(config.weight)
        self.bias: Optional[nn.Parameter] = None
        self.q_weight: Optional[QStorage] = None

    def enable_sync(self):
        """启用同步操作，并递归启用所有子量化器的同步"""
        self.sync = True
        if hasattr(self.input_quantizer, 'enable_sync'):
            self.input_quantizer.enable_sync()
        if hasattr(self.weight_quantizer, 'enable_sync'):
            self.weight_quantizer.enable_sync()

    @validate_call(config=dict(arbitrary_types_allowed=True))
    def setup(self, linear: nn.Linear):
        self.weight = linear.weight
        self.bias = linear.bias
        self.weight_quantizer.init_weight(QStorage(QDType.FLOAT, value=linear.weight.detach()), self.bias)

        for hook_id, hook in linear._forward_pre_hooks.items():
            with_kwargs = hook_id in linear._forward_pre_hooks_with_kwargs
            self.register_forward_pre_hook(hook, with_kwargs=with_kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with QStorage.set_value_float_type(x.dtype):
            x = self.input_quantizer(x)
            weight = self.weight_quantizer(x)
        return F.linear(x, weight, self.bias)

    def deploy(self):
        fake_quantizer = qir.AutoFakeQuantLinear.create(
            self.input_quantizer.get_q_param(),
            self.weight_quantizer.get_q_param(),
            self.weight_quantizer.get_q_storage(),
            self.bias
        )

        for hook in self._forward_pre_hooks.values():
            if isinstance(hook, qir.HookIR):
                fake_quantizer = hook.wrapper_module(fake_quantizer)

        return fake_quantizer

    def support_distributed(self) -> bool:
        """
        判断是否支持分布式
        通过检查 input_quantizer 和 weight_quantizer 是否都支持分布式来判断
        
        Returns:
            bool: 是否支持分布式
        """
        input_support = self.input_quantizer.support_distributed()
        weight_support = self.weight_quantizer.support_distributed()
        return input_support and weight_support
