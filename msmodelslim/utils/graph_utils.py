# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.

import torch.nn as nn
from .dag_utils.torch_dag_adapter import TorchDAGAdapter


def extract_dag(
        model: nn.Module,
        dummy_input=None,
        hook_nodes=None,    
        anti_method=None):
    return TorchDAGAdapter(model, dummy_input, hook_nodes=hook_nodes, anti_method=anti_method)