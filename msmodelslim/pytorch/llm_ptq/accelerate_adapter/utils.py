# Copyright Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.

import gc
from functools import lru_cache
from typing import Optional

import torch
from accelerate.utils import OffloadedWeightsLoader

HF_HOOK = '_hf_hook'
WEIGHTS_MAP = 'weights_map'
OLD_HOOK = 'old_hook'
DEVICE_CPU = 'cpu'


def judge_model_with_accelerate(model: torch.nn.Module) -> bool:
    """

    """
    for _, mod in model.named_modules():
        if judge_module_with_accelerate(mod):
            return True
    return False


def judge_module_with_accelerate(module: torch.nn.Module) -> bool:
    """

    """
    return hasattr(module, HF_HOOK)


def get_offloaded_dataset(model: torch.nn.Module) -> Optional[OffloadedWeightsLoader]:
    """
    if accelerate is on and offload to disk, return dataset, else return None
    """

    def check_weights_loader(hook):
        return (
                hasattr(hook, WEIGHTS_MAP)
                and getattr(hook, WEIGHTS_MAP) is not None
                and isinstance(getattr(hook, WEIGHTS_MAP).dataset, OffloadedWeightsLoader)
                and getattr(hook, WEIGHTS_MAP).dataset.save_folder
        )

    for _, mod in model.named_modules():
        if not judge_module_with_accelerate(mod):
            continue
        hook = getattr(mod, HF_HOOK)
        if hasattr(hook, OLD_HOOK):
            hook = hook.old_hook
        if check_weights_loader(hook):
            return getattr(hook, WEIGHTS_MAP).dataset
    return None


@lru_cache(maxsize=1)
def is_npu_available():
    try:
        import torch_npu
    except ImportError:
        return False

    return torch.npu.is_available()


@lru_cache(maxsize=1)
def is_cuda_available():
    return hasattr(torch, 'cuda') and torch.cuda.is_available()


def clear_device_cache(garbage_collection=False):
    if garbage_collection:
        gc.collect()

    if is_npu_available():
        torch.npu.empty_cache()
    elif is_cuda_available():
        torch.cuda.empty_cache()
