# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional

from msmodelslim.core.const import DeviceType
from ..quant_service.dataset_loader_infra import DatasetLoaderInfra
from ...core.runner.pipeline_interface import PipelineInterface


class BaseAnalysisService(ABC):
    """Base class for model analysis services"""

    def __init__(self, dataset_loader: DatasetLoaderInfra):
        self.dataset_loader = dataset_loader

    @abstractmethod
    def analyze(self,
                model_adapter: PipelineInterface,
                patterns: List[str],
                analysis_config: Optional[Dict[str, Any]] = None,
                device: DeviceType = DeviceType.NPU):
        """
        Analyze model layers based on given patterns
        
        Args:
            model_adapter: The model to analyze
            patterns: List of layer name patterns to analyze (e.g., ['*linear*', 'attention.*'])
            analysis_config: Configuration for analysis method
            device: device

        """
        raise NotImplementedError

    @abstractmethod
    def export_results(self, result: Any, top_k: int = 10):
        """
        export analysis results in service-specific format
        
        Args:
            result: AnalysisResult containing layer scores and metadata
            top_k: Number of top layers to display
        """
        raise NotImplementedError
