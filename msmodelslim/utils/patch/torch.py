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

import torch
from torch import nn


def _is_torch_nn_module_has_get_submodule() -> bool:
    """
    判断torch.nn.Module是否具有get_submodule方法
    """
    return hasattr(nn.Module, "get_submodule")


def _is_torch_nn_module_has_set_submodule() -> bool:
    """
    判断torch.nn.Module是否具有set_submodule方法
    """
    return hasattr(nn.Module, "set_submodule")


def _torch_nn_module_get_submodule(self, name: str) -> nn.Module:
    """
    如果torch.nn.Module不具有get_submodule方法，则添加一个
    """
    tokens = name.split('.')
    cur_mod = self
    for s in tokens:
        cur_mod = getattr(cur_mod, s, None)
    return cur_mod


def _torch_nn_module_set_submodule(self, name: str, submodule: nn.Module):
    """
    如果torch.nn.Module不具有set_submodule方法，则添加一个
    """
    tokens = name.split('.')
    sub_tokens = tokens[:-1]
    cur_mod = self
    for s in sub_tokens:
        cur_mod = getattr(cur_mod, s)
    setattr(cur_mod, tokens[-1], submodule)


def _is_torch_has_get_default_device() -> bool:
    """
    判断torch是否具有get_default_device方法
    """
    return hasattr(torch, "get_default_device")


_TORCH_DEFAULT_DEVICE = torch.device("cpu")


def _torch_set_default_device(device: torch.device):
    """
    设置torch的默认设备
    """
    global _TORCH_DEFAULT_DEVICE
    _TORCH_DEFAULT_DEVICE = device


def _torch_get_default_device() -> torch.device:
    """
    获取torch的默认设备
    """
    return _TORCH_DEFAULT_DEVICE


def patch_torch():
    """
    如果torch.nn.Module不具有get_submodule方法，则添加一个
    """
    if not _is_torch_nn_module_has_get_submodule():
        nn.Module.get_submodule = _torch_nn_module_get_submodule
    if not _is_torch_nn_module_has_set_submodule():
        nn.Module.set_submodule = _torch_nn_module_set_submodule
    if not _is_torch_has_get_default_device():
        original_set_default_device = torch.set_default_device

        def _set_default_device(device):
            _torch_set_default_device(device)
            return original_set_default_device(device)

        torch.set_default_device = _set_default_device
        torch.get_default_device = _torch_get_default_device
