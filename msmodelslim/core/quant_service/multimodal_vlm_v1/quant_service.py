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

from pathlib import Path
from typing import Optional, Literal, Any, List

import torch

from msmodelslim.core.const import RunnerType, DeviceType
from msmodelslim.core.quant_service import DatasetLoaderInfra
from msmodelslim.core.quant_service import KeyInfoPersistenceInfra
from msmodelslim.core.runner.layer_wise_runner import LayerWiseRunner
from msmodelslim.core.runner.pipeline_interface import PipelineInterface
from msmodelslim.core.runner.optional_interface import LayerWiseOffloadOptionalInterface
from msmodelslim.utils.exception import SchemaValidateError
from msmodelslim.utils.logging import get_logger, logger_setter
from msmodelslim.utils.seed import seed_all
from msmodelslim.core.context import ContextManager, IContextFactory
from .quant_config import MultimodalVLMModelslimV1QuantConfig
from ..interface import BaseQuantConfig, IQuantService, QuantServiceConfig


class MultimodalVLMModelslimV1QuantServiceConfig(QuantServiceConfig):
    """multimodal_vlm_modelslim_v1 量化服务配置，用于插件选择与 QuantService 初始化。"""
    apiversion: Literal["multimodal_vlm_modelslim_v1"] = "multimodal_vlm_modelslim_v1"


@logger_setter(
    prefix='msmodelslim.core.quant_service.multimodal_vlm_modelslim_v1')  # 4-level: msmodelslim.core.quant_service.multimodal_vlm_modelslim_v1
class MultimodalVLMModelslimV1QuantService(IQuantService):
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

    def __init__(
        self,
        quant_service_config: MultimodalVLMModelslimV1QuantServiceConfig,
        dataset_loader: DatasetLoaderInfra,
        context_factory: Optional[IContextFactory] = None,
        debug_info_persistence: Optional[KeyInfoPersistenceInfra] = None,
        **kwargs,
    ):
        """
        Initialize multimodal VLM quantization service.

        Args:
            quant_service_config: MultimodalVLMModelslimV1QuantServiceConfig.
            dataset_loader: DatasetLoaderInfra（用于加载数据集）.
            context_factory: Optional context factory for dependency injection.
            debug_info_persistence: Optional context persistence for saving debug info.
        """
        self.quant_service_config = quant_service_config
        self.dataset_loader = dataset_loader
        self.context_factory = context_factory
        self.debug_info_persistence = debug_info_persistence

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

        if device_indices is not None:
            get_logger().warning(
                "Specifying device indices is not supported in %s quant_service. "
                "Device indices will be ignored.",
                self.backend_name
            )

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

        offload_device = "cpu"
        if isinstance(model_adapter, LayerWiseOffloadOptionalInterface):
            preferred_offload = model_adapter.get_layer_wise_offload_device()
            if isinstance(preferred_offload, str) and preferred_offload:
                if preferred_offload in ("cpu", "meta"):
                    offload_device = preferred_offload
                else:
                    get_logger().warning(
                        f"Invalid offload device {preferred_offload} from model adapter, fallback to 'cpu'. "
                        "Supported: ['cpu', 'meta']."
                    )

        runner = LayerWiseRunner(adapter=model_adapter, offload_device=offload_device)
        ctx = self.context_factory.create()
        get_logger().info(f"Created runner LayerWiseRunner successfully")
        with ContextManager(ctx=ctx):
            # Add all processors
            for process_cfg in final_process_cfg:
                runner.add_processor(processor_cfg=process_cfg)

            # Run quantization
            runner.run(calib_data=dataset, device=device)
            get_logger().info(f"==========QUANTIZATION: END==========")

        # Save context if persistence is provided
        if self.debug_info_persistence is not None:
            get_logger().info(f"==========SAVE CONTEXT DEBUG INFO==========")
            try:
                self.debug_info_persistence.save_from_context(ctx=ctx)
            except Exception as e:
                get_logger().warning(f"Failed to save debug info: {e}")


def get_plugin():
    """
    获取 multimodal_vlm_modelslim_v1 量化服务插件（返回配置类与组件类，由框架完成注册）。
    Returns:
        (MultimodalVLMModelslimV1QuantServiceConfig, MultimodalVLMModelslimV1QuantService) 元组
    """
    return MultimodalVLMModelslimV1QuantServiceConfig, MultimodalVLMModelslimV1QuantService
