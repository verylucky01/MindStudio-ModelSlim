# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.

from typing import Any, Callable, Dict, Optional

# 全局缓存
_cache: Dict[str, Any] = {}


def load_cached(key: str, init_func: Callable, args=None, kwargs=None) -> Any:
    args = args or tuple()
    kwargs = kwargs or dict()
    if key not in _cache:
        _cache[key] = init_func(*args, **kwargs)
    return _cache[key]


def clear_cache(key: Optional[str] = None):
    if key is None:
        _cache.clear()
    else:
        _cache.pop(key, None)
