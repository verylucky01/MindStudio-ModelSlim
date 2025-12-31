#  -*- coding: utf-8 -*-
#  Copyright (c) 2024-2024 Huawei Technologies Co., Ltd.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#  http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

"""
Anti-outlier common utilities
"""

__all__ = [
    # Fused Linear
    'VirtualVModuleFromQKVFused',
    'VirtualVModuleFromKVFused',
    # Smooth Components
    'HookManager',
    'StatsCollector',
    'SubgraphRegistry',
    'StatKey',
    # Schema Types
    'SmoothQuantContext',
    'SmoothQuantConfig',
    'IterSmoothContext',
    'IterSmoothConfig',
    'FlexSmoothQuantContext',
    'FlexSmoothQuantConfig',
    'FlexAWQSSZContext',
    'FlexAWQSSZConfig',
    'SmoothContext',
    # Scale Computation
    'IterSmoothScaleCalculator',
    'FlexSmoothScaleCalculator',
    'FlexAWQSSZScaleCalculator',
    'BaseScaleCalculator',
    'validate_and_process_tensors',
    'compute_weight_scale',
    'compute_multi_weight_scale',
    'apply_smooth_scale_shift',
    'prepare_mqga_parameters',
    'reduce_scales_for_mqga_mean',
    'reduce_scales_for_mqga_max',
    'MQGAScaleParams',
    # Subgraph Fusion
    'SubgraphFusionFactory',
    'SubgraphFusionStrategy',
    'OVSubgraphFusion',
    'UpDownSubgraphFusion',
    'LinearLinearSubgraphFusion',
    'NormLinearSubgraphFusion',
]

from .fused_linear import VirtualVModuleFromQKVFused, VirtualVModuleFromKVFused
from .smooth_components import (
    HookManager,
    StatsCollector,
    SubgraphRegistry,
    StatKey
)
from .smooth_types import (
    SmoothQuantContext,
    SmoothQuantConfig,
    IterSmoothContext,
    IterSmoothConfig,
    FlexSmoothQuantContext,
    FlexSmoothQuantConfig,
    FlexAWQSSZContext,
    FlexAWQSSZConfig,
    SmoothContext
)
from .scale_computation import (
    BaseScaleCalculator,
    FlexAWQSSZScaleCalculator,
    FlexSmoothScaleCalculator,
    IterSmoothScaleCalculator,
    MQGAScaleParams,
    apply_smooth_scale_shift,
    compute_multi_weight_scale,
    compute_weight_scale,
    prepare_mqga_parameters,
    reduce_scales_for_mqga_max,
    reduce_scales_for_mqga_mean,
    validate_and_process_tensors
)
from .subgraph_fusion import (
    LinearLinearSubgraphFusion,
    NormLinearSubgraphFusion,
    OVSubgraphFusion,
    SubgraphFusionFactory,
    SubgraphFusionStrategy,
    UpDownSubgraphFusion
)

