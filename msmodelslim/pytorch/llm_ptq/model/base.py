#  -*- coding: utf-8 -*-
#  Copyright (c) 2024-2024 Huawei Technologies Co., Ltd.
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

from typing import Dict, Optional, List, Any, Type, TYPE_CHECKING, Tuple

import torch
import torch.nn as nn
from transformers import PreTrainedModel

from msmodelslim import logger

if TYPE_CHECKING:
    from msmodelslim.pytorch.llm_ptq.anti_outlier.config import AntiOutlierConfig


class ModelAdapter:
    def __init__(self, model):
        self.model = model

    def get_norm_linear_subgraph(self,
                                 cfg: 'AntiOutlierConfig',
                                 dummy_input: Optional[torch.Tensor] = None,
                                 norm_class: Optional[List[Type[nn.Module]]] = None) -> Dict[str, List[str]]:
        """获取Norm->Linear子图"""
        _ = self
        return {}

    def modify_smooth_args(self,
                           cfg: 'AntiOutlierConfig',
                           norm_name: str,
                           linear_names: str,
                           args: List[Any],
                           kwargs: Dict[str, Any]) -> Tuple[List[Any], Dict[str, Any]]:
        """修改异常值抑制参数"""
        _ = self
        return args, kwargs


class DefaultModelAdapter(ModelAdapter):
    def __init__(self, model):
        super().__init__(model)

    def get_norm_linear_subgraph(self,
                                 cfg: 'AntiOutlierConfig',
                                 dummy_input: Optional[torch.Tensor] = None,
                                 norm_class: Optional[List[Type[nn.Module]]] = None) -> Dict[str, List[str]]:
        """获取Norm->Linear子图"""

        logger.debug(f"try to get norm linear subgraph using dag for {type(self.model)}")

        if dummy_input is None or norm_class is None:
            raise ValueError("dummy_input and norm_class must be provided if using default model adapter")

        from msmodelslim.pytorch.llm_ptq.anti_outlier.graph_utils import extract_dag

        dag = extract_dag(self.model,
                          dummy_input,
                          hook_nodes=norm_class,
                          anti_method=cfg.anti_method)

        norm_linear_subgraph = dag.get_norm_linear_subgraph()

        if cfg.anti_method in ['m4', 'm6']:
            linear_linear_subgraph = dag.get_linear_linear_subgraph()
            norm_linear_subgraph.update(linear_linear_subgraph)

        del dag

        return norm_linear_subgraph


class ModelAdapterRegistry:
    """
    模型适配器注册表，用于管理不同模型类型的适配器
    """
    _registry: Dict[str, type] = {}

    @classmethod
    def register(cls, model_type: str):
        """
        注册模型类型和对应的适配器类
        
        参数:
        model_type (str): 模型类型的名称
        """

        def decorator(adapter_class):
            cls._registry[model_type] = adapter_class
            return adapter_class

        return decorator

    @classmethod
    def get_adapter(cls, model: PreTrainedModel) -> Optional['ModelAdapter']:
        """
        根据模型获取对应的适配器实例
        
        参数:
        model (PreTrainedModel): 预训练模型
        
        返回:
        Optional[ModelAdapter]: 对应的适配器实例或 None
        """
        if hasattr(model, 'config') and hasattr(model.config, 'model_type'):
            model_type = model.config.model_type
            if model_type and model_type in cls._registry:
                logger.debug(f"find model adapter {cls._registry[model_type]} for {model_type}")
                return cls._registry[model_type](model)
            logger.debug(f"can not find model adapter for {model_type}, use default model adapter")
        return DefaultModelAdapter(model)
