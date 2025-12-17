#  -*- coding: utf-8 -*-
#  Copyright (c) 2025-2025 Huawei Technologies Co., Ltd.
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

from pathlib import Path
from typing import Optional, Literal, Any, List

import torch

from msmodelslim.core.const import RunnerType, DeviceType
from msmodelslim.app.quant_service.base import BaseQuantService
from msmodelslim.app.quant_service import DatasetLoaderInfra
from msmodelslim.core.runner.layer_wise_runner import LayerWiseRunner
from msmodelslim.core.runner.pipeline_interface import PipelineInterface
from msmodelslim.utils.exception import SchemaValidateError
from msmodelslim.utils.logging import get_logger, logger_setter
from msmodelslim.utils.seed import seed_all
from .quant_config import MultimodalVLMModelslimV1QuantConfig
from ..interface import BaseQuantConfig


@logger_setter(prefix='msmodelslim.app.quant_service.multimodal_vlm_modelslim_v1')
class MultimodalVLMModelslimV1QuantService(BaseQuantService):
    """
    Quantization service for multimodal vision-language models (V1 framework).
    
    Features:
    - Layer-wise loading and processing (memory efficient)
    - Automatic MoE fusion layer conversion
    - Multi-modal calibration dataset support
    - Compatible with msmodelslim quant command
    
    Supported models:
    - Qwen3-VL-MoE
    - Other multimodal VLM models (extensible)
    """
    
    backend_name: str = "multimodal_vlm_modelslim_v1"

    def __init__(self, dataset_loader: DatasetLoaderInfra):
        """
        Initialize multimodal VLM quantization service.
        
        Args:
            dataset_loader: Dataset loader for multimodal data.
        """
        super().__init__(dataset_loader)

    @staticmethod
    def _choose_runner_type(quant_config: MultimodalVLMModelslimV1QuantConfig,
                            model_adapter: PipelineInterface) -> Literal[
        RunnerType.MODEL_WISE, RunnerType.LAYER_WISE]:
        """
        Choose runner type based on config.
        
        For multimodal VLM models, we default to LAYER_WISE for memory efficiency.
        
        Args:
            quant_config: Quantization configuration
            model_adapter: Model adapter
        
        Returns:
            Runner type (MODEL_WISE or LAYER_WISE)
        """
        if quant_config.spec.runner == RunnerType.MODEL_WISE:
            get_logger().info("Model-wise runner detected, using model-wise pipeline.")
            return RunnerType.MODEL_WISE

        if quant_config.spec.runner == RunnerType.LAYER_WISE:
            get_logger().info("Layer-wise runner detected, using layer-wise pipeline.")
            return RunnerType.LAYER_WISE

        # Default to layer-wise for memory efficiency
        get_logger().info("Runner type not detected, defaulting to layer-wise pipeline (recommended for VLM).")
        return RunnerType.LAYER_WISE

    def quantize(self, quant_config: BaseQuantConfig, model_adapter: Any, save_path: Optional[Path] = None,
                 device: DeviceType = DeviceType.NPU, device_indices: Optional[List[int]] = None):
        """
        Main quantization entry point.
        
        Args:
            quant_config: Base quantization config (will be converted to MultimodalVLMV1QuantConfig)
            model_adapter: Model adapter implementing PipelineInterface
            save_path: Path to save quantized model
            device: Device for quantization (NPU or CPU)
        """
        # Validate inputs
        if not isinstance(quant_config, BaseQuantConfig):
            raise SchemaValidateError("task is not a BaseTask",
                                      action="Please make sure the task is a BaseTask")
        if not isinstance(model_adapter, PipelineInterface):
            raise SchemaValidateError("model_adapter must be a PipelineInterface",
                                      action="Please make sure the model_adapter is a PipelineInterface")
        if save_path is not None and not isinstance(save_path, Path):
            raise SchemaValidateError("save_path must be a Path or None",
                                      action="Please make sure the save_path is a Path or None")
        if not isinstance(device, DeviceType):
            raise SchemaValidateError("device must be a DeviceType",
                                      action="Please make sure the device is a DeviceType")

        return self.quant_process(
            MultimodalVLMModelslimV1QuantConfig.from_base(quant_config),
            model_adapter,
            save_path,
            device,
            device_indices
        )

    def quant_process(self,
                      quant_config: MultimodalVLMModelslimV1QuantConfig,
                      model_adapter: PipelineInterface,
                      save_path: Optional[Path],
                      device: DeviceType = DeviceType.NPU,
                      device_indices: Optional[List[int]] = None
                      ):
        """
        Core quantization process.
        
        Steps:
        1. Set random seed
        2. Load dataset (multimodal images)
        3. Automatically insert MoE converter if needed
        4. Create runner (layer-wise by default)
        5. Add processors (anti-outlier, quant, save, etc.)
        6. Run quantization
        
        Args:
            quant_config: Multimodal VLM quantization config
            model_adapter: Model adapter
            save_path: Save path
            device: Device
        """
        common_seed = 42
        seed_all(seed=common_seed, mode=True)

        if device == DeviceType.NPU:
            # Enable binary compilation for NPU
            torch.npu.set_compile_mode(jit_compile=False)

        get_logger().info(f"==========QUANTIZATION: Prepare Dataset==========")
        
        dataset_path = quant_config.spec.dataset
        # Set default_text to dataset_loader
        self.dataset_loader.default_text = quant_config.spec.default_text
        dataset = self.dataset_loader.get_dataset_by_name(dataset_path)
        get_logger().info(f"Prepared dataset from {dataset_path} successfully")

        final_process_cfg = quant_config.spec.process.copy()
        
        # Note: MoE conversion is now handled automatically in model_adapter during layer loading
        # No need for separate MoeConverterProcessor
        
        if save_path is not None:
            get_logger().info(f"==========QUANTIZATION: Prepare Persistence==========")
            for save_cfg in quant_config.spec.save:
                save_cfg.set_save_directory(save_path)

            # Register save processors
            final_process_cfg += quant_config.spec.save
            get_logger().info(f"Prepared persistence to {save_path} successfully")

        get_logger().info(f"==========QUANTIZATION: Run Quantization==========")
        
        if quant_config.spec.runner != "layer_wise":
            get_logger().warning(
                f"runner for multimodal_vlm_modelslim_v1 is not layer_wise, will be converted to layer_wise.")
        
        runner = LayerWiseRunner(adapter=model_adapter)

        get_logger().info(f"Created runner LayerWiseRunner successfully")

        # Add all processors
        for process_cfg in final_process_cfg:
            runner.add_processor(processor_cfg=process_cfg)

        # Run quantization
        runner.run(calib_data=dataset, device=device)
        get_logger().info(f"==========QUANTIZATION: END==========")