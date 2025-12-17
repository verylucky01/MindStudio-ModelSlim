# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
from enum import Enum
from pathlib import Path
from typing import List

from msmodelslim.core.const import DeviceType
from msmodelslim.model import IModelFactory
from msmodelslim.utils.exception import SchemaValidateError, UnsupportedError
from msmodelslim.utils.exception_decorator import exception_catcher
from msmodelslim.utils.logging import logger_setter, get_logger
from msmodelslim.utils.validation.conversion import (
    convert_to_readable_dir
)
from msmodelslim.utils.validation.value import validate_str_length
from ..analysis_service import BaseAnalysisService
from ..analysis_service.pipeline_interface import PipelineInterface


class AnalysisMetrics(str, Enum):
    """Enumeration of valid analysis metrics"""
    STD = 'std'
    QUANTILE = 'quantile'
    KURTOSIS = 'kurtosis'


@logger_setter('msmodelslim.app.analysis.application')
class LayerAnalysisApplication:
    """Application for analyzing model layer sensitivity"""

    def __init__(
            self,
            analysis_service: BaseAnalysisService,
            model_factory: IModelFactory,
    ):
        self.analysis_service = analysis_service
        self.model_factory = model_factory

    @exception_catcher
    def analyze(self,
                model_type: str,
                model_path: str,
                patterns: List[str],
                device: DeviceType = DeviceType.NPU,
                metrics: AnalysisMetrics = AnalysisMetrics.KURTOSIS,
                calib_dataset: str = 'boolq.jsonl',
                topk: int = 15,
                trust_remote_code: bool = False):
        """
        Run layer analysis on a model
        
        Args:
            model_type: Type of the model (e.g., 'Qwen2.5-7B-Instruct')
            model_path: Path to the model
            patterns: List of layer name patterns to analyze (e.g., ['*linear*', 'attention.*'])
            device: Device to run analysis on
            metrics: Analysis metrics ('quantile' 、 'std'、 'kurtosis')
            calib_dataset: Dataset path for calibration
            topk: Number of top layers to output for disable_names
            trust_remote_code: Whether to trust remote code
        """
        # Validate string inputs with length checks
        str_params = [
            ("model_type", model_type),
            ("model_path", model_path),
            ("calib_dataset", calib_dataset)
        ]
        for param_name, value in str_params:
            if not isinstance(value, str):
                raise SchemaValidateError(f"{param_name} must be a string, but got {type(value)}")
            validate_str_length(input_str=value, str_name=param_name)

        # Validate inputs
        model_path = convert_to_readable_dir(model_path)
        if not isinstance(model_path, Path):
            raise SchemaValidateError(f"model_path must be a Path, but got {type(model_path)}")
        if not isinstance(patterns, list):
            raise SchemaValidateError(f"pattern must be a list, but got {type(patterns)}")
        if not isinstance(device, DeviceType):
            raise SchemaValidateError(f"device must be a DeviceType")
        if not isinstance(metrics, AnalysisMetrics):
            raise SchemaValidateError(f"metrics must be a AnalysisMetrics")
        if not isinstance(calib_dataset, str):
            raise SchemaValidateError(f"calib_dataset must be a string, but got {type(calib_dataset)}")
        # Validate file format - only support .json and .jsonl
        if not (calib_dataset.endswith('.json') or calib_dataset.endswith('.jsonl')):
            raise SchemaValidateError(
                f'Unsupported file format: {calib_dataset}. '
                'Only .json and .jsonl formats are supported',
                action='Please provide a file with .json or .jsonl extension'
            )
        if not isinstance(topk, int) or topk <= 0:
            raise SchemaValidateError(f"topk must be a integer greater than 0, but got {topk}")
        if not isinstance(trust_remote_code, bool):
            raise SchemaValidateError(f"trust_remote_code must be a bool")

        # Log parameters
        get_logger().info(f'Layer analysis with following parameters:')
        get_logger().info(f"model_type: {model_type}")
        get_logger().info(f"model_path: {model_path}")
        get_logger().info(f"pattern: {patterns}")
        get_logger().info(f"device: {device}")
        get_logger().info(f"metrics: {metrics}")
        get_logger().info(f"calib_dataset: {calib_dataset}")
        get_logger().info(f"topk: {topk}")
        get_logger().info(f"trust_remote_code: {trust_remote_code}")

        return self._analyze(
            model_type, model_path, patterns, device,
            metrics, calib_dataset, topk, trust_remote_code
        )

    def _analyze(self,
                 model_type: str,
                 model_path: Path,
                 patterns: List[str],
                 device: DeviceType,
                 metrics: AnalysisMetrics,
                 calib_dataset: str,
                 topk: int,
                 trust_remote_code: bool):
        """Internal analysis implementation"""
        # Run analysis
        get_logger().info(f"===========RUN ANALYSIS===========")

        # Create analysis config from parameters
        analysis_config = {
            'metrics': metrics.value,
            'calib_dataset': calib_dataset,
            'method_params': {}
        }

        model_adapter = self.model_factory.create(
            model_type, model_path, trust_remote_code
        )
        if not isinstance(model_adapter, PipelineInterface):
            raise UnsupportedError(f'Model adapter {model_adapter.__class__.__name__} does NOT support analyze',
                                   action='Please implement PipelineInterface for model analyzing')

        result = self.analysis_service.analyze(
            device=device,
            model_adapter=model_adapter,
            patterns=patterns,
            analysis_config=analysis_config
        )

        if result is None:
            get_logger().info(f"===========ANALYSIS COMPLETE===========")
            return result

        # export results using service-specific formatter
        self.analysis_service.export_results(result, topk)

        get_logger().info(f"===========ANALYSIS COMPLETE===========")
        return result
