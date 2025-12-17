# Copyright Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.

from typing import Callable, Mapping, MutableMapping

import torch


def handle_lazy_tensor(dic: MutableMapping) -> None:
    for key in dic:
        if isinstance(dic[key], LazyTensor):
            dic[key] = dic[key].value


def get_tensor_size(tensor: torch.Tensor) -> int:
    return tensor.numel() * tensor.element_size()


class LazyTensor:
    def __init__(self, func: Callable[..., torch.Tensor], tensor: torch.Tensor = None, **kwargs):
        self._func = func
        self._kwargs = kwargs

        if tensor is not None:
            self._size = get_tensor_size(tensor)
            return

        tensor = self._func(**self._kwargs)
        self._size = get_tensor_size(tensor)

    @property
    def value(self):
        return self._func(**self._kwargs).cpu().contiguous()

    @property
    def size(self):
        return self._size
