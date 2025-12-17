# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
import functools
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Callable, Optional
import fnmatch

import torch
import torch.nn as nn
from tqdm import tqdm

from msmodelslim.utils.logging import get_logger

logger = get_logger('msmodelslim.app.analysis_service.analysis')


class AnalysisTargetMatcher:
    """Helper class to match layers based on patterns and types"""
    
    @staticmethod
    def get_linear_conv_layers(model: nn.Module) -> List[str]:
        """Get all linear and convolutional layer names"""
        layer_names = []
        for name, module in model.named_modules():
            if isinstance(module, (nn.Linear, nn.modules.linear.NonDynamicallyQuantizableLinear, nn.Conv2d)):
                layer_names.append(name)
        return layer_names
    
    @staticmethod
    def filter_layers_by_patterns(layer_names: List[str], patterns: List[str]) -> List[str]:
        """Filter layer names by patterns"""
        if not patterns or patterns == ['*']:
            return layer_names
            
        filtered = []
        for layer_name in layer_names:
            for pattern in patterns:
                if fnmatch.fnmatch(layer_name, pattern):
                    filtered.append(layer_name)
                    break
        return filtered


class LayerAnalysisMethod(ABC):
    """Abstract base class for layer analysis methods"""
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Name of the analysis method"""
        pass
    
    @abstractmethod
    def get_hook(self) -> Callable:
        """Get the hook function to collect data during model inference"""
        pass
    
    @abstractmethod
    def compute_score(self, layer_data: Dict[str, Any]) -> float:
        """Compute analysis score for a layer given collected data"""
        pass
    
    def analyze_layers(self, layer_stats: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Analyze all layers and return scores"""
        layer_scores = []
        for name in tqdm(layer_stats, desc=f"Computing {self.name} scores"):
            score = self.compute_score(layer_stats[name])
            layer_scores.append({'name': name, 'score': score})
        return layer_scores


class QuantileAnalysisMethod(LayerAnalysisMethod):
    """Quantile-based layer analysis method"""

    def __init__(self, sample_step: int = 100):
        self.sample_step = sample_step

    @property
    def name(self) -> str:
        return "quantile"

    @staticmethod
    def get_quantile_score(act: torch.Tensor, device: torch.device):
        """Helper method to compute quantile score from activation tensor"""
        act_max = torch.max(torch.abs(act))
        act = act.to(device)
        sorted_act = torch.sort(act)[0]
        sorted_act = sorted_act.to("cpu")
        number = len(sorted_act)
        number_1_4 = number // 4
        number_3_4 = number_1_4 * 3
        range_param = 2 * act_max / 254 / (sorted_act[number_3_4] - sorted_act[number_1_4] + 1e-10)
        return range_param.item()

    def compute_score(self, layer_data: Dict[str, Any]) -> float:
        """Compute quantile score for the layer"""
        tensor_data = torch.cat(layer_data['tensor']).view(-1).float()
        device = layer_data['device']
        score = QuantileAnalysisMethod.get_quantile_score(tensor_data, device)
        return score

    def get_hook(self) -> Callable:
        """Get hook function for collecting activation data"""
        def activation_hook(module, input_tensor, output_tensor, layer_name, stats_dict):
            if isinstance(input_tensor, tuple):
                input_tensor = input_tensor[0]

            # Flatten and sort the input tensor
            flattened = input_tensor.reshape(-1)
            sorted_tensor = torch.sort(flattened)[0]

            # Sample the tensor
            if sorted_tensor.numel() < self.sample_step:
                sampled = sorted_tensor
            else:
                sampled = sorted_tensor[self.sample_step // 2::self.sample_step].view(-1, 1)

            # Store data
            if layer_name not in stats_dict:
                stats_dict[layer_name] = {'tensor': [sampled.to('cpu')], 'device': input_tensor.device}
            else:
                stats_dict[layer_name]['tensor'].append(sampled.to('cpu'))

        return activation_hook


class StdAnalysisMethod(LayerAnalysisMethod):
    """Standard deviation-based layer analysis method"""
    
    @property
    def name(self) -> str:
        return "std"

    def compute_score(self, layer_data: Dict[str, Any]) -> float:
        """Compute std-based score for the layer"""
        abs_max = max(abs(layer_data['t_max']), abs(layer_data['t_min']))

        # 防止除零：如果标准差为0，返回abs_max或0
        std_value = layer_data['std']
        if std_value == 0:
            return abs_max.item() if abs_max > 0 else 0.0

        range_param = abs_max / std_value
        return range_param.item()

    def get_hook(self) -> Callable:
        def activation_hook(module, input_tensor, output_tensor, layer_name, stats_dict):
            if isinstance(input_tensor, tuple):
                input_tensor = input_tensor[0]

            tensor_float = input_tensor.float()
            hidden_dim = tensor_float.shape[-1]
            reshaped = tensor_float.reshape(-1, hidden_dim).detach()

            tensor_max = torch.max(reshaped, dim=0)[0]
            tensor_min = torch.min(reshaped, dim=0)[0]

            if layer_name not in stats_dict:
                stats_dict[layer_name] = {}

            # Store shift (center point)
            stats_dict[layer_name]['shift'] = (tensor_max + tensor_min) / 2

            # Update global max/min
            global_max = torch.max(reshaped)
            global_min = torch.min(reshaped)

            if 't_max' in stats_dict[layer_name]:
                stats_dict[layer_name]['t_max'] = torch.max(stats_dict[layer_name]['t_max'], global_max)
                stats_dict[layer_name]['t_min'] = torch.min(stats_dict[layer_name]['t_min'], global_min)
            else:
                stats_dict[layer_name]['t_max'] = global_max
                stats_dict[layer_name]['t_min'] = global_min

            # Update standard deviation
            tensor_std = torch.std(reshaped - stats_dict[layer_name]['shift'])
            if 'std' in stats_dict[layer_name]:
                stats_dict[layer_name]['std'] = torch.max(stats_dict[layer_name]['std'], tensor_std)
            else:
                stats_dict[layer_name]['std'] = tensor_std

        return activation_hook


class KurtosisAnalysisMethod(LayerAnalysisMethod):
    """Kurtosis-based layer analysis method"""
    
    def __init__(self, sample_step: int = 100):
        self.sample_step = sample_step
    
    @property
    def name(self) -> str:
        return "kurtosis"

    def compute_score(self, layer_data: Dict[str, Any]) -> float:
        """Compute quantile score for the layer"""
        tensor_data = torch.cat(layer_data['tensor']).view(-1).float()
        score = kurtosis(tensor_data)
        return score.item()

    def get_hook(self) -> Callable:
        def activation_hook(module, input_tensor, output_tensor, layer_name, stats_dict):
            if isinstance(input_tensor, tuple):
                input_tensor = input_tensor[0]

            # Flatten and sort the input tensor
            flattened = input_tensor.reshape(-1)
            sorted_tensor = torch.sort(flattened)[0]

            # Sample the tensor
            if sorted_tensor.numel() < self.sample_step:
                sampled = sorted_tensor
            else:
                sampled = sorted_tensor[self.sample_step // 2::self.sample_step].view(-1, 1)

            # Store data
            if layer_name not in stats_dict:
                stats_dict[layer_name] = {'tensor': [sampled.to('cpu')], 'device': input_tensor.device}
            else:
                stats_dict[layer_name]['tensor'].append(sampled.to('cpu'))

        return activation_hook


class AnalysisMethodFactory:
    """Factory for creating analysis methods"""
    
    _methods = {
        'quantile': QuantileAnalysisMethod,
        'std': StdAnalysisMethod,
        'kurtosis': KurtosisAnalysisMethod,
    }
    
    @classmethod
    def create_method(cls, method_name: str, **kwargs) -> LayerAnalysisMethod:
        """Create an analysis method by name"""
        if method_name not in cls._methods:
            supported = list(cls._methods.keys())
            raise ValueError(f"Unsupported analysis method: {method_name}. Supported methods: {supported}")
        
        method_class = cls._methods[method_name]
        return method_class(**kwargs)
    
    @classmethod
    def register_method(cls, method_name: str, method_class: type):
        """Register a new analysis method"""
        if not issubclass(method_class, LayerAnalysisMethod):
            raise TypeError("Method class must inherit from LayerAnalysisMethod")
        cls._methods[method_name] = method_class
    
    @classmethod
    def get_supported_methods(cls) -> List[str]:
        """Get list of supported method names"""
        return list(cls._methods.keys()) 


def kurtosis(x: torch.Tensor, dim=None, keepdim=False) -> float:
    """
    Compute the kurtosis of a tensor along a given dimension.
    """
    if dim is not None:
        mean = x.mean(dim=dim, keepdim=True)
        std = x.std(dim=dim, unbiased=False, keepdim=True)
    else:
        mean = x.mean()
        std = x.std(unbiased=False)
    z = (x - mean) / (std + 1e-10)
    kurt = (z.pow(4).mean(dim=dim, keepdim=keepdim) - 3)

    return kurt