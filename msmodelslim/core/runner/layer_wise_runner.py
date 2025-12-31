#  -*- coding: utf-8 -*-
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
from typing import List, Optional, Any

import torch
from torch import nn

from msmodelslim.core.base.protocol import DataUnit
from msmodelslim.core.const import DeviceType
from msmodelslim.core.runner.generated_runner import GeneratedRunner, get_input_datas
from msmodelslim.core.runner.pipeline_interface import PipelineInterface
from msmodelslim.processor import LoadProcessorConfig, AutoProcessorConfig
from msmodelslim.utils.logging import logger_setter, get_logger


@logger_setter()
class LayerWiseRunner(GeneratedRunner):
    """
        "Module-Processor-Dataset" Mode Runner
        WARNING: Module will be REMOVED after handing, do NOT yield module more than once

        Calibrate quant perception:
        1、Only perceive the processors in front
        2、Perceive the calibration results of all processors in front
    """

    def __init__(self, adapter: PipelineInterface, offload_device: str = 'meta'):
        super().__init__(adapter)
        self.offload_device = offload_device

    def preprocess_processor(self, processor_list: List[AutoProcessorConfig], model: nn.Module,
                             device: DeviceType = DeviceType.NPU):
        
        if device == DeviceType.NPU:
            # 从当前设备获取实际设备索引
            current_device_idx = torch.npu.current_device()
            device_str = f"npu:{current_device_idx}"
            get_logger().debug(f"Using device index {current_device_idx} from torch.npu.current_device()")
        else:
            device_str = device

        # 逐层上传推理设备
        processor_list.insert(0,
                              LoadProcessorConfig(
                                  device=device_str,
                                  mode="load", post_offload=True
                              ))
        # 逐层释放推理设备内存
        processor_list.append(LoadProcessorConfig(device=self.offload_device, mode="offload", cleanup=True))

    def run(self, model: nn.Module = None, calib_data: Optional[List[Any]] = None,
            device: DeviceType = DeviceType.NPU, device_indices: Optional[List[int]] = None):
        _ = get_input_datas(self.adapter, calib_data, DeviceType.CPU)

        if model is None:
            get_logger().info('Start to init model')
            model = self.adapter.init_model(device=DeviceType.CPU)
            get_logger().info('Init model success')

        if device == DeviceType.NPU and device_indices:
            torch.npu.set_device(f"npu:{device_indices[0]}")

        processor_list = self.process_config_list.copy()
        self.preprocess_processor(processor_list, model, device=device)

        data_recorder = DataUnit(None, None)
        process_unit = self.build_process_unit(processor_list,
                                               model=model,
                                               adapter=self.adapter,
                                               calib_data=calib_data,
                                               data_recorder=data_recorder)

        self.generated_schedule(process_unit, data_recorder)
