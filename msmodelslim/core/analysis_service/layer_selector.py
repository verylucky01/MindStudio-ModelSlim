# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
import functools
from itertools import groupby
from typing import List, Dict, Any, Optional

import torch
import torch.nn as nn
from tqdm import tqdm

from msmodelslim.core.const import DeviceType
from msmodelslim.utils.exception import SchemaValidateError
from msmodelslim.utils.logging import logger_setter, get_logger, clean_output
from .analysis_methods import AnalysisMethodFactory, AnalysisTargetMatcher
from .base import BaseAnalysisService
from .pipeline_interface import PipelineInterface
from msmodelslim.core.quant_service.dataset_loader_infra import DatasetLoaderInfra


class AnalysisResult:
    """Analysis result containing layer scores and metadata"""

    def __init__(self, layer_scores: List[Dict[str, Any]], method: str, patterns: List[str]):
        self.layer_scores = layer_scores  # List of {'name': str, 'score': float}
        self.method = method
        self.patterns = patterns

    def get_sorted_layers(self, reverse: bool = True) -> List[Dict[str, Any]]:
        """Get layers sorted by score"""
        return sorted(self.layer_scores, key=lambda x: x['score'], reverse=reverse)


def get_tokenized_data(tokenizer, calib_list, device,
                       input_ids_name='input_ids',
                       attention_mask_name='attention_mask'):
    """Get tokenized calibration data"""
    tokenized_data = []
    for input_text in calib_list:
        inputs = tokenizer(input_text, return_tensors='pt', padding=True).to(device)
        tokenized_data.append([inputs.data[input_ids_name], inputs.data[attention_mask_name]])
    return tokenized_data


@logger_setter()
class LayerSelectorAnalysisService(BaseAnalysisService):
    """Analysis service for layer sensitivity evaluation using various methods"""

    def __init__(self, dataset_loader: DatasetLoaderInfra):
        super().__init__(dataset_loader)

    def analyze(self,
                model_adapter: PipelineInterface,
                patterns: List[str],
                analysis_config: Optional[Dict[str, Any]] = None,
                device: DeviceType = DeviceType.NPU) -> AnalysisResult:
        """
        Analyze layer sensitivity based on patterns and configuration
        
        Args:
            model_adapter: The model to analyze
            patterns: List of layer name patterns to analyze (e.g., ['*linear*', 'attention.*'])
            analysis_config: Configuration including:
                - method: Analysis method name ('quantile' or 'std')
                - calib_dataset: Dataset name for calibration
                - method_params: Parameters for the analysis method
            device: device
                
        Returns:
            AnalysisResult containing layer scores and metadata
        """
        if not isinstance(patterns, list):
            raise SchemaValidateError("patterns must be a list",
                                      action='Please provide patterns as a list of strings')

        # Set default configuration
        config = analysis_config or {}
        metrics_name = config.get('metrics')
        calib_dataset_name = config.get('calib_dataset')
        method_params = config.get('method_params')

        get_logger().info(f"==========ANALYSIS: Starting Layer Analysis==========")
        get_logger().info(f"Analysis metrics: {metrics_name}")
        get_logger().info(f"Layer patterns: {patterns}")

        model = model_adapter.load_model(device)

        # Initialize NPU compilation if needed
        if device is DeviceType.NPU:
            torch.npu.set_compile_mode(jit_compile=False)

        # Prepare calibration data
        calib_data = self._prepare_calibration_data(model_adapter, calib_dataset_name, device)

        # Get target layers
        target_layers = self._get_target_layers(model.model, patterns)
        if not target_layers:
            get_logger().warning("No layers found matching the specified patterns")
            get_logger().info(f"==========ANALYSIS: Analysis Complete==========")
            return None

        get_logger().info(f"Found {len(target_layers)} target layers matching patterns")

        # Create analysis method
        analysis_method = AnalysisMethodFactory.create_method(metrics_name, **method_params)
        get_logger().info(f"Using analysis metrics: {analysis_method.name}")

        # Run analysis
        layer_scores = self._run_analysis(model.model, target_layers, analysis_method, calib_data)

        # Create result
        result = AnalysisResult(layer_scores, metrics_name, patterns)

        get_logger().info(f"==========ANALYSIS: Analysis Complete==========")
        return result

    def export_results(self, result: AnalysisResult, disable_level: int = 15):
        """Export analysis results in YAML format"""
        self._print_analysis_results(result, disable_level)

    def _prepare_calibration_data(self, model_adapter: PipelineInterface,
                                  calib_dataset_name: Optional[str],
                                  device: DeviceType = DeviceType.NPU):
        """Prepare calibration data for analysis"""
        if calib_dataset_name is None:
            get_logger().warning("No calibration dataset specified. Analysis may be less accurate.")
            return None

        get_logger().info(f"Loading calibration dataset: {calib_dataset_name}")
        dataset = self.dataset_loader.get_dataset_by_name(calib_dataset_name)
        calib_data = model_adapter.handle_dataset(dataset, device=device)
        get_logger().info(f"Loaded {len(calib_data)} calibration samples")
        return calib_data

    def _get_target_layers(self, model: nn.Module, patterns: List[str]) -> List[str]:
        """Get target layer names based on patterns"""
        # Get all linear and conv layers
        all_layers = AnalysisTargetMatcher.get_linear_conv_layers(model)

        # Filter by patterns
        target_layers = AnalysisTargetMatcher.filter_layers_by_patterns(all_layers, patterns)

        return target_layers

    def _run_analysis(self,
                      model: nn.Module,
                      target_layers: List[str],
                      analysis_method,
                      calib_data) -> List[Dict[str, Any]]:
        """Run the analysis on target layers"""
        get_logger().info(f"==========ANALYSIS: Collecting Layer Statistics==========")

        # Collect statistics
        layer_stats = {}
        hooks = []

        # Register hooks
        hook_func = analysis_method.get_hook()
        for name, module in model.named_modules():
            if name in target_layers and isinstance(module, nn.Linear):
                hook = module.register_forward_hook(
                    functools.partial(hook_func, layer_name=name, stats_dict=layer_stats)
                )
                hooks.append(hook)

        # Run model with calibration data
        if calib_data is not None:
            get_logger().info(f"Running model inference on {len(calib_data)} calibration samples")
            for data in tqdm(calib_data, desc="Processing calibration data"):
                with torch.no_grad():
                    if isinstance(data, (tuple, list)):
                        model(*data)
                    elif isinstance(data, dict):
                        model(**data)
                    else:
                        raise NotImplementedError(f"Unsupported data type: {type(data)}")
        else:
            get_logger().warning("No calibration data available. Skipping model inference.")

        # Remove hooks
        for hook in hooks:
            hook.remove()

        if not layer_stats:
            get_logger().warning(
                "No statistics collected. This may be due to no calibration data "
                "or incompatible layer types."
            )
            return []

        # Compute layer scores
        get_logger().debug(f"==========ANALYSIS: Computing Layer Scores==========")
        layer_scores = analysis_method.analyze_layers(layer_stats)

        # Log scores
        sorted_scores = sorted(layer_scores, key=lambda x: x['score'], reverse=True)
        get_logger().debug(f"Layer analysis scores ({analysis_method.name} method):")
        for item in sorted_scores:
            get_logger().debug(f"  {item['name']}: {item['score']:.4f}")

        return layer_scores

    def _print_analysis_results(self, result: AnalysisResult, disable_level: int = 15):
        """Print layer selector analysis results with quantization focus"""
        sorted_layers = result.get_sorted_layers(reverse=True)
        layer_groups = [list(group) for key, group in groupby(sorted_layers, key=lambda x: x['score'])]

        # Log basic analysis info
        get_logger().info(f"=== Layer Analysis Results ({result.method} method) ===")
        get_logger().info(f"Patterns analyzed: {result.patterns}")
        get_logger().info(f"Total layers analyzed: {len(result.layer_scores)}")
        get_logger().info("Layer Sensitivity Scores (higher score = more sensitive to quantization):")
        get_logger().info("-" * 80)

        # Get top K layers for disable_names and flatten the groups
        if disable_level <= len(layer_groups) and disable_level >= 0:
            selected_groups = layer_groups[:disable_level]
        else:
            selected_groups = layer_groups

        # Flatten the selected groups to get individual layer dictionaries
        display_layers = []
        for group in selected_groups:
            display_layers.extend(group)

        # Log layer scores
        for i, layer_info in enumerate(display_layers, 1):
            get_logger().info(f"{i:3d}. {layer_info['name']:50s} | Score: {layer_info['score']:8.4f}")

        get_logger().info("-" * 80)

        # Log summary
        get_logger().info(f"Top {len(display_layers)} most sensitive layers selected for disable_names")

        # Print clean YAML format for easy copying
        get_logger().info("")  # Empty line for separation
        get_logger().info("=== YAML Format for quantization ===")
        get_logger().info("")  # Empty line for separation

        # Use clean output context to print YAML without log prefixes
        with clean_output():
            get_logger().info(f"top {len(display_layers)}:")
            for layer_info in display_layers:
                get_logger().info(f"  - '{layer_info['name']}'")

        get_logger().info("")  # Empty line for separation
        get_logger().info("=== End of YAML Format ===")
