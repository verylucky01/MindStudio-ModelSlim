#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""
-------------------------------------------------------------------------
This file is part of the MindStudio project.
Copyright (c) 2025 Huawei Technologies Co.,Ltd.

MindStudio is licensed under Mulan PSL v2.
You can use this software according to the terms and conditions of the Mulan PSL v2.
You may obtain a copy of Mulan PSL v2 at:

         http://license.coscl.org.cn/MulanPSL2

THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
See the Mulan PSL v2 for more details.
-------------------------------------------------------------------------
"""


from typing import List, Literal, Dict

import torch
import torch.nn as nn
from pydantic import field_validator

from msmodelslim.ir.qal.qregistry import QABCRegistry
from msmodelslim.core.base.protocol import BatchProcessRequest
from msmodelslim.processor.base import AutoProcessorConfig, AutoSessionProcessor
from msmodelslim.utils.exception import SchemaValidateError
from msmodelslim.utils.exception import UnsupportedError
from msmodelslim.utils.logging import get_logger
from .quarot_interface import QuaRotInterface, RotSide, get_rotate_command
from .laos_online import LAOSOnlineRotationProcessor
from ..common.quarot_utils import fuse_ln_linear, rotate_linear, is_power_of_two, bake_mean_into_linear


class QuaRotProcessorConfig(AutoProcessorConfig):
    type: Literal["quarot"] = "quarot"
    online: bool = False
    block_size: int = -1
    down_proj_online_layers: List[int] = []
    max_tp_size: int = 4

    @field_validator('max_tp_size')
    @classmethod
    def validate_max_tp_size(cls, v):
        """校验 max_tp_size：必须大于等于1且为2的幂"""
        if v < 1 or not is_power_of_two(v):
            raise SchemaValidateError(f"max_tp_size must be a positive power of 2 or equal to 1, got {v}")
        return v

    @field_validator('block_size')
    @classmethod
    def validate_block_size(cls, v):
        """校验 block_size：取值范围为-1或大于0且为2的幂的整数"""
        if v == -1:
            return v
        if v <= 0 or not is_power_of_two(v):
            raise SchemaValidateError(f"block_size must be -1 or a positive power of 2, got {v}")
        return v


@QABCRegistry.register(dispatch_key=QuaRotProcessorConfig, abc_class=AutoSessionProcessor)
class QuaRotProcessor(AutoSessionProcessor):
    def __init__(self, model: nn.Module, config: QuaRotProcessorConfig, adapter: QuaRotInterface, **kwargs) -> None:
        super().__init__(model)
        self.config = config
        self.model = model
        self.adapter = adapter
        self.fused_map = {}
        self.bake_names = []
        self.rotate_commands = []
        if not isinstance(adapter, QuaRotInterface):
            raise UnsupportedError(f'{adapter.__class__.__name__} does not support QuaRot',
                                   action='Please provide a valid model adapter '
                                          'which implements QuaRotInterface')
        if self.config.online:
            self.online_processor = LAOSOnlineRotationProcessor(model, config, adapter)

    def support_distributed(self) -> bool:
        return True

    def is_data_free(self) -> bool:
        return True

    def pre_run(self) -> None:
        pre_run_fused_ln, self.fused_map = self.adapter.get_ln_fuse_map()
        pre_run_bake_names, self.bake_names = self.adapter.get_bake_names()
        pre_run_pairs, self.rotate_pairs = self.adapter.get_rotate_map(block_size=self.config.block_size)
        pre_run_commands = get_rotate_command(pre_run_pairs)
        self._fuse_norm(pre_run_fused_ln)
        self._bake_mean(pre_run_bake_names)
        self._rotate(pre_run_commands)
        self.rotate_commands = get_rotate_command(self.rotate_pairs)
        if self.config.online:
            self.online_processor.pre_run()

    def preprocess(self, request: BatchProcessRequest) -> None:
        prefix = request.name
        prefix = f"{prefix}." if prefix != "" else ""
        fused_map = self._filter_fused_map(prefix)
        bake_names = self._filter_bake_names(prefix)
        rotate_commands = self._filter_commands(prefix)
        self._fuse_norm(fused_map)
        self._bake_mean(bake_names)
        self._rotate(rotate_commands)
        if self.config.online:
            self.online_processor.preprocess(request)
    
    def post_run(self) -> None:
        self._fuse_norm(self.fused_map)
        self.fused_map = {}
        self._bake_mean(self.bake_names)
        self.bake_names = []
        self._rotate(self.rotate_commands)
        self.rotate_commands = []
        if self.config.online:
            self.online_processor.post_run()

    def _filter_fused_map(self, prefix: str) -> Dict[str, str]:
        res = {}
        for key, value in self.fused_map.items():
            select = False
            if isinstance(value, list):
                for v in value:
                    if v.startswith(prefix):
                        select = True
            else:
                if value.startswith(prefix):
                    select = True
            if select:
                res[key] = value
        self.fused_map = {k: v for k, v in self.fused_map.items() if k not in res}
        return res

    def _filter_bake_names(self, prefix: str):
        res = [name for name in self.bake_names if name.startswith(prefix)]
        for name in res:
            self.bake_names.remove(name)
        return res

    def _filter_commands(self, prefix: str):
        res = [command for command in self.rotate_commands if command.target.startswith(prefix)]
        for command in res:
            self.rotate_commands.remove(command)
        return res

    def _fuse_norm(self, fused_map: Dict[str, str]):
        for key, value in fused_map.items():
            get_logger().debug(f"start to fuse layer norm and linear: {key} and {value}")
            layernorms = []
            if isinstance(key, list) or isinstance(key, tuple):
                for k in key:
                    layernorms.append(self.model.get_submodule(k))
            else:
                layernorms.append(self.model.get_submodule(key))
            linears = []

            if isinstance(value, list) or isinstance(value, tuple):
                for v in value:
                    linears.append(self.model.get_submodule(v))
            else:
                linears.append(self.model.get_submodule(value))
            try:
                fuse_ln_linear(layernorms, linears)
            except UnsupportedError as e:
                raise UnsupportedError("fuse layer norm and linear error!",
                                    action=f"Please check the {key} and {value} size!") from e
            get_logger().debug(f"successfully fuse layer norm and linear: {key} and {value}")

    def _bake_mean(self, bake_names: List[str]):
        for name in bake_names:
            get_logger().debug(f"start to bake mean into linear: {name}")
            mod = self.model.get_submodule(name)
            if isinstance(mod, torch.nn.Linear):
                bake_mean_into_linear(mod)
                get_logger().debug(f"successfully bake mean into linear: {name}")
            else:
                raise UnsupportedError("bake mean into linear error!",
                                    action=f"Please check the {name} type and model adapter implementation!")

    def _rotate(self, rotate_commands: List[str]):
        for command in rotate_commands:
            get_logger().debug(f"start to {command.side.value} rotate linear: {command.target}")
            mod = self.model.get_submodule(command.target)
            try:
                rotate_linear(mod, command.rot, command.side == RotSide.RIGHT)
            except UnsupportedError as e:
                raise UnsupportedError(f"{command.side.value} rotate linear error!",
                                    action=f"Please check whether the {command.target} size is equal \
                                    to the rotate matrix size: {command.rot.shape[0]}!") from e
            get_logger().debug(f"{command.side.value} rotate linear success: {command.target}")
