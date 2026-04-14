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

from typing import Any, Callable, Dict, List

import torch
import torch.nn as nn

from msmodelslim.processor.analysis.binary_operator_layer_wise.metrics.base import LayerWiseAnalysisMethod
from msmodelslim.utils.logging import get_logger


logger = get_logger()


class MSELayerWiseAnalysisMethod(LayerWiseAnalysisMethod):
    """
    mse_layer_wise 指标实现：
    - 针对每个 DecoderLayer，收集其内部所有 nn.Linear 的浮点与量化输出；
    - 逐 Linear 计算 MSE，再在 DecoderLayer 粒度做均值。
    """

    def __init__(self, adapter: object = None):
        self.adapter = adapter

    @property
    def name(self) -> str:
        return "mse_layer_wise"

    def get_hook(self) -> Callable:
        def activation_hook(module, input_tensor, output_tensor, layer_name: str, stats_dict: Dict[str, Any]):
            if isinstance(output_tensor, tuple):
                tensor = output_tensor[0]
            else:
                tensor = output_tensor

            tensor = tensor.detach().to("cpu")
            stats_dict[layer_name] = {"output": tensor}

        return activation_hook

    def get_target_decoder_layers(self, model: nn.Module, prefix: str) -> List[str]:
        decoder_layers: List[str] = []
        for name, module in model.named_modules(prefix=prefix):
            if ".layers." in name and any(
                sub_name.startswith(name + ".") for sub_name, _ in model.named_modules(prefix=prefix)
            ):
                decoder_layers.append(name)
        unique_layers: List[str] = []
        for n in decoder_layers:
            if n not in unique_layers:
                unique_layers.append(n)
        return unique_layers

    def get_linear_layers_for_decoder(self, model: nn.Module, decoder_layer_name: str) -> List[str]:
        linear_layers: List[str] = []
        for name, module in model.named_modules(prefix=decoder_layer_name):
            if isinstance(module, nn.Linear):
                linear_layers.append(name)
        return linear_layers

    def compute_decoder_layer_scores(
        self,
        float_stats: Dict[str, Any],
        quant_stats: Dict[str, Any],
        target_decoder_layers: List[str],
    ) -> List[Dict[str, Any]]:
        decoder_scores: List[Dict[str, Any]] = []

        for decoder_name in target_decoder_layers:
            linear_layer_names = [
                layer_name
                for layer_name in float_stats.keys()
                if layer_name.startswith(decoder_name + ".") or layer_name == decoder_name
            ]

            mse_values: List[torch.Tensor] = []
            for layer_name in linear_layer_names:
                float_entry = float_stats.get(layer_name)
                quant_entry = quant_stats.get(layer_name)
                if not float_entry or not quant_entry:
                    continue

                float_out = float_entry.get("output")
                quant_out = quant_entry.get("output")
                if float_out is None or quant_out is None:
                    continue

                try:
                    if float_out.shape != quant_out.shape:
                        logger.warning(
                            "Skip MSE for linear layer %s due to shape mismatch: float_out=%s, quant_out=%s",
                            layer_name, tuple(float_out.shape), tuple(quant_out.shape),
                        )
                        continue
                    mse = torch.nn.functional.mse_loss(float_out.float(), quant_out.float())
                    if torch.isnan(mse).item():
                        logger.warning("Skip MSE for linear layer %s due to mse is nan.", layer_name)
                        continue
                    mse_values.append(mse)
                except Exception as exc:  # pragma: no cover - defensive branch
                    logger.warning("Failed to compute MSE for linear layer %s. ErrorMsg: %s", layer_name, exc)

            if not mse_values:
                logger.warning("Skip MSE for DecoderLayer %s duo to mse_values is empty.", decoder_name)
                continue

            score = torch.stack(mse_values).mean().item()
            decoder_scores.append({"name": decoder_name, "score": score})

        return decoder_scores

    def _matches(self, module: nn.Module) -> bool:
        return False
