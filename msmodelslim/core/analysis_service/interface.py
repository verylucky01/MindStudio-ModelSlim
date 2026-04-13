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
from abc import ABC, abstractmethod
from typing import Any, Dict, List

from pydantic import BaseModel, Field

from msmodelslim.core.const import DeviceType
from msmodelslim.core.runner.pipeline_interface import PipelineInterface


class AnalysisConfig(BaseModel):
    """分析服务入参：指标类型、校准数据集、层匹配模式及方法参数等。"""

    metrics: str = Field(
        ...,
        description="分析指标名，如 std / quantile / kurtosis / attention_mse / mse_model_wise",
    )
    calib_dataset: str = Field(..., description="校准数据集名称，用于前向收集激活")
    patterns: List[str] = Field(default_factory=lambda: ["*"], description="层名匹配模式，支持 fnmatch")


class AnalysisResult(BaseModel):
    """分析结果数据：层分数列表及方法、patterns 等元数据。"""

    layer_scores: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="List of {'name': str, 'score': float}",
    )
    method: str = Field(..., description="分析方法名，如 std / kurtosis")
    patterns: List[str] = Field(default_factory=list, description="层匹配模式")


class IAnalysisService(ABC):
    """Abstract base class for model analysis services"""

    @abstractmethod
    def analyze(
        self,
        model_adapter: PipelineInterface,
        analysis_config: AnalysisConfig,
        device: DeviceType = DeviceType.NPU,
    ) -> AnalysisResult:
        """
        Analyze model layers based on given configuration.

        Args:
            model_adapter: The model to analyze
            analysis_config: 分析配置（metrics / calib_dataset / patterns）
            device: 运行设备
        """
        ...
        