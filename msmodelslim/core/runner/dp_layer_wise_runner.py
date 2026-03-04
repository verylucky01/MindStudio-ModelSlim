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
from typing import List, Optional, Any
import os

from torch import nn
import torch.distributed as dist
import torch.multiprocessing as mp

from msmodelslim.core.const import DeviceType
from msmodelslim.core.runner.layer_wise_runner import LayerWiseRunner
from msmodelslim.core.runner.pipeline_interface import PipelineInterface
from msmodelslim.processor import AutoProcessorConfig
from msmodelslim.processor.base import AutoSessionProcessor
from msmodelslim.utils.config import msmodelslim_config
from msmodelslim.utils.distributed import find_free_port, setup_distributed
from msmodelslim.utils.logging import logger_setter, get_logger, set_logger_level
from msmodelslim.core.runner.generated_runner import get_input_datas
from msmodelslim.utils.exception import UnsupportedError
from msmodelslim.core.context import ContextManager, get_current_context



@logger_setter()
class DPLayerWiseRunner(LayerWiseRunner):
    """
    Distributed Data Parallel Layer-Wise Runner for multi-device quantization.
    This runner inherits from LayerWiseRunner and adds distributed execution support.
    
    It handles:
    1. Setting up distributed environment
    2. Launching multiple processes
    3. Running quantization on each device
    """

    def __init__(self, adapter: PipelineInterface, offload_device: str = 'meta',
                 backend: str = 'hccl'):
        """
        Initialize DPLayerWiseRunner.
        
        Args:
            adapter: PipelineInterface instance
            offload_device: Device to offload model layers to (default: 'meta')
            backend: Distributed backend ('hccl' for NPU, 'nccl' for CUDA)
        """
        super().__init__(adapter, offload_device=offload_device)
        self.backend = backend

    @staticmethod
    def convert_to_distributed_config_if_needed(processor_cfg: AutoProcessorConfig) -> AutoProcessorConfig:
        """
        Convert AscendV1Config to DistributedAscendV1Config if needed for distributed saving.
        
        Args:
            processor_cfg: Processor configuration to check and convert
        
        Returns:
            Converted processor configuration (or original if no conversion needed)
        """
        from msmodelslim.core.quant_service.modelslim_v1.save.ascendv1 import AscendV1Config
        from msmodelslim.core.quant_service.modelslim_v1.save.ascendv1_distributed import DistributedAscendV1Config
        if isinstance(processor_cfg, AscendV1Config) and not isinstance(processor_cfg, DistributedAscendV1Config):
            # Convert to DistributedAscendV1Config
            distributed_cfg = DistributedAscendV1Config(
                type="ascendv1_saver_distributed",
                save_directory=processor_cfg.save_directory,
                part_file_size=processor_cfg.part_file_size,
                ext=processor_cfg.ext
            )
            get_logger().info(
                f"Converted AscendV1Config to DistributedAscendV1Config for distributed saving"
            )
            return distributed_cfg
        
        return processor_cfg

    def distributed_worker(self, rank: int, world_size: int, device_indices: List[int],
                          model: Optional[nn.Module], calib_data: Optional[List[Any]], 
                          device: DeviceType, master_port: int = 29500,
                          shared_ctx=None):
        """
        Worker function for distributed execution.
        
        Args:
            rank: Process rank (0-based index in the process group)
            world_size: Total number of processes
            device_indices: List of device indices to use (e.g., [0, 1, 2, 3])
            model: Model to quantize (optional)
            calib_data: Calibration data (optional)
            device: Target device type
            master_port: Master port for distributed communication
            shared_ctx: Shared context instance
        """
        try:
            with ContextManager(ctx=shared_ctx):
                set_logger_level(msmodelslim_config.env_vars.log_level)

                # Get the actual device index for this rank
                actual_device_idx = device_indices[rank]
                
                # Setup distributed environment
                # rank is used for process group communication, actual_device_idx is used for device selection
                setup_distributed(rank, world_size, self.backend, device_index=actual_device_idx, master_port=master_port)
                
                get_logger().info(
                    f"Rank {rank}/{world_size} initialized on device {actual_device_idx} "
                    f"(device index {actual_device_idx}) with backend {self.backend}"
                )

            
                # Initialize model in distributed environment
                _ = get_input_datas(self.adapter, calib_data, DeviceType.CPU)
                
                if model is None:
                    get_logger().info('Start to init model in distributed environment')
                    model = self.adapter.init_model(device=DeviceType.CPU)
                    get_logger().info('Init model success in distributed environment')
                
                # Check if all processors support distributed execution (only rank 0 performs check)
                unsupported_processors = self._check_distributed_support(self.process_config_list, model)
                
                if unsupported_processors:
                    # Found unsupported processors, raise error
                    unsupported_names = [str(p) for p in unsupported_processors]
                    error_msg = (
                        f"The following processors do not support distributed quantization: {unsupported_names}. "
                        f"Please check the processor configuration."
                    )
                    get_logger().error(error_msg)
                    raise UnsupportedError(error_msg)
                
                # Execute quantization (based on layer_wise_runner.run)
                processor_list = self.process_config_list.copy()
                self.preprocess_processor(processor_list, model, device=device)
                
                from msmodelslim.core.base.protocol import DataUnit
                data_recorder = DataUnit(None, None)
                process_unit = self.build_process_unit(
                    processor_list,
                    model=model,
                    adapter=self.adapter,
                    calib_data=calib_data,
                    data_recorder=data_recorder
                )
                
                self.generated_schedule(process_unit, data_recorder)
            
        except Exception as e:
            get_logger().error(f"Error in rank {rank}: {e}")
            raise
        finally:
            if dist.is_initialized():
                dist.destroy_process_group()

    def run(self, model: nn.Module = None, calib_data: Optional[List[Any]] = None,
            device: DeviceType = DeviceType.NPU, device_indices: Optional[List[int]] = None):
        """
        Run distributed quantization.
        
        This method sets up the distributed environment and launches multiple processes
        for parallel quantization across multiple devices.
        
        Args:
            model: Model to quantize (optional)
            calib_data: Calibration data (optional)
            device: Target device type
            device_indices: List of device indices to use (e.g., [0, 1, 2, 3])
        """
        if device_indices is None or len(device_indices) <= 1:
            get_logger().warning("Number of devices <= 1, falling back to single-device execution")
            return super().run(model=model, calib_data=calib_data, device=device, device_indices=device_indices)
        
        world_size = len(device_indices)

        get_logger().info(
            f"Starting distributed execution with {world_size} devices: {device_indices}. "
        )
        
        try:
            # Set multiprocessing start method
            mp.set_start_method('spawn', force=True)
            
            # Find available port in main process
            if 'MASTER_PORT' not in os.environ:
                master_port = find_free_port()
                os.environ['MASTER_PORT'] = str(master_port)
                get_logger().info(f"Main process: Using port {master_port} for distributed quantization")
            else:
                master_port = int(os.environ['MASTER_PORT'])
                get_logger().info(
                    f"Main process: Using existing MASTER_PORT {master_port} for distributed quantization"
                )
            
            shared_ctx = get_current_context()

            # Start distributed execution
            mp.spawn(
                self.distributed_worker,
                args=(world_size, device_indices, model, calib_data, device, master_port, shared_ctx),
                nprocs=world_size,
                join=True
            )
            return None
        except Exception as e:
            get_logger().error(f"Failed to start distributed execution: {e}")
            raise

    def add_processor(self, processor_cfg: AutoProcessorConfig, append: bool = True):
        """
        Add a processor configuration to the runner.
        
        For DPLayerWiseRunner, this method automatically converts AscendV1Config
        to DistributedAscendV1Config for distributed saving support.
        
        Args:
            processor_cfg: Processor configuration to add
            append: If True, append to the end; if False, insert at the beginning
        """
        # Convert AscendV1Config to DistributedAscendV1Config if needed
        converted_cfg = self.convert_to_distributed_config_if_needed(processor_cfg)
        
        if append:
            self.process_config_list.append(converted_cfg)
        else:
            self.process_config_list.insert(0, converted_cfg)

    def _check_distributed_support(self, processor_list: List[AutoProcessorConfig], 
                                    model: nn.Module) -> List[AutoSessionProcessor]:
        """
        Check which processors support distributed execution.
        
        Args:
            processor_list: List of processor configurations to check
            model: Model instance (needed to create processor instances)
        
        Returns:
            List of processors that do NOT support distributed execution
        """
        # Create processor instances to check distributed support
        processors: List[AutoSessionProcessor] = []
        for processor_config in processor_list:
            processor = AutoSessionProcessor.from_config(model, processor_config, self.adapter)
            processors.append(processor)
        
        # Find processors that do not support distributed execution
        unsupported_processors = [processor for processor in processors if not processor.support_distributed()]
        
        return unsupported_processors
