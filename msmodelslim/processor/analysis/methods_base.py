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
import fnmatch
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List

import torch.nn as nn


class AnalysisTargetMatcher(ABC):
    """抽象基类：按模型结构匹配目标层。子类只需实现 _matches(module)，基类统一做 named_modules 遍历。"""

    def get_target_layers(self, model: nn.Module, prefix: str = "") -> List[str]:
        """遍历 model.named_modules()，收集满足 _matches(module) 的层名（唯一循环）。"""
        _all_target_layers = []
        for name, module in model.named_modules(prefix=prefix):
            if self._matches(module):
                _all_target_layers.append(name)
        return _all_target_layers

    @staticmethod
    def filter_layers_by_patterns(layer_names: List[str], patterns: List[str]) -> List[str]:
        """按 patterns 过滤层名（支持 fnmatch）。"""
        if not patterns or patterns == ['*']:
            return layer_names
        filtered = []
        for layer_name in layer_names:
            for pattern in patterns:
                if fnmatch.fnmatch(layer_name, pattern):
                    filtered.append(layer_name)
                    break
        return filtered

    @abstractmethod
    def _matches(self, module: nn.Module) -> bool:
        """当前 module 是否算作目标层。"""
        ...


class LayerAnalysisMethod(ABC):

    @property
    @abstractmethod
    def name(self) -> str:
        """Name of the analysis method"""
        ...

    @abstractmethod
    def get_hook(self) -> Callable:
        """Get the hook function to collect data during model inference."""
        ...


class UnaryAnalysisMethod(LayerAnalysisMethod):
    """一元分析方法基类"""

    @abstractmethod
    def compute_score(self, layer_data: Dict[str, Any]) -> float:
        """Compute analysis score for a layer given collected data"""
        ...


class BinaryAnalysisMethod(LayerAnalysisMethod):
    """二元分析方法基类"""

    @abstractmethod
    def compute_score(self, layer_data_before: Dict[str, Any], layer_data_after: Dict[str, Any]) -> float:
        """Compute analysis score for a layer given collected data before and after"""
        ...


def _get_method_classes():
    """Lazy import to avoid circular dependency."""
    from .unary_analysis_methods.quantile import QuantileAnalysisMethod
    from .unary_analysis_methods.std import StdAnalysisMethod
    from .unary_analysis_methods.kurtosis import KurtosisAnalysisMethod
    from .binary_analysis_methods.attention_mse import AttentionMSEAnalysisMethod
    return {
        'quantile': QuantileAnalysisMethod,
        'std': StdAnalysisMethod,
        'kurtosis': KurtosisAnalysisMethod,
        'attention_mse': AttentionMSEAnalysisMethod,
    }


class AnalysisMethodFactory:
    """Factory for creating analysis methods"""
    
    _methods = None

    @classmethod
    def _get_methods(cls):
        if cls._methods is None:
            cls._methods = _get_method_classes()
        return cls._methods
    
    @classmethod
    def create_method(cls, method_name: str, **kwargs) -> LayerAnalysisMethod:
        """Create an analysis method by name"""
        methods = cls._get_methods()
        if method_name not in methods:
            supported = list(methods.keys())
            raise ValueError(f"Unsupported analysis method: {method_name}. Supported methods: {supported}")
        method_class = methods[method_name]
        return method_class(**kwargs)

    @classmethod
    def register_method(cls, method_name: str, method_class: type):
        """Register a new analysis method"""
        if not issubclass(method_class, LayerAnalysisMethod):
            raise TypeError("Method class must inherit from LayerAnalysisMethod")
        cls._get_methods()[method_name] = method_class

    @classmethod
    def get_supported_methods(cls) -> List[str]:
        """Get list of supported method names"""
        return list(cls._get_methods().keys())
        