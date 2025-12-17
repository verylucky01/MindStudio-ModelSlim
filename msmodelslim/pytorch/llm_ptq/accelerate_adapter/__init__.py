# Copyright Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.

__all__ = [
    'replace_device_align_hook_if_needed',
    'move_update_weight_hook_if_need',
    'get_state_dict_copy',
    'clear_unused_module',
    'PrepareWeight',

    'get_offloaded_dataset',

    'LazyTensor',
    'handle_lazy_tensor'
]

from .hook_adapter import (replace_device_align_hook_if_needed,
                           move_update_weight_hook_if_need,
                           get_state_dict_copy,
                           clear_unused_module,
                           PrepareWeight)
from .utils import get_offloaded_dataset
from .lazy_handler import LazyTensor, handle_lazy_tensor
