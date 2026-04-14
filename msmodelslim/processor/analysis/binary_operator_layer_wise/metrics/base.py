#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""
-------------------------------------------------------------------------
This file is part of the MindStudio project.
Copyright (c) 2026 Huawei Technologies Co.,Ltd.

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

from abc import abstractmethod
from typing import Any, Dict, List

import torch.nn as nn

from msmodelslim.processor.analysis.methods_base import (
    AnalysisTargetMatcher,
    LayerAnalysisMethod,
)


class LayerWiseAnalysisMethod(LayerAnalysisMethod, AnalysisTargetMatcher):
    """Base class for layer-operator (DecoderLayer-level) analysis methods."""

    @abstractmethod
    def get_target_decoder_layers(self, model: nn.Module, prefix: str) -> List[str]:
        """Return names of decoder layers under given prefix."""
        ...

    @abstractmethod
    def get_linear_layers_for_decoder(self, model: nn.Module, decoder_layer_name: str) -> List[str]:
        """Return linear sub-module names inside a given decoder layer."""
        ...

    @abstractmethod
    def compute_decoder_layer_scores(
        self,
        float_stats: Dict[str, Any],
        quant_stats: Dict[str, Any],
        target_decoder_layers: List[str],
    ) -> List[Dict[str, Any]]:
        """Compute scores at decoder-layer granularity based on per-linear stats."""
        ...
