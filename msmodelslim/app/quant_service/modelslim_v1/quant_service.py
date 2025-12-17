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
import shutil
from pathlib import Path
from typing import Optional, Literal, List

import torch

from msmodelslim.app.quant_service.base import BaseQuantService
from msmodelslim.app.quant_service.dataset_loader_infra import DatasetLoaderInfra
from msmodelslim.core.const import RunnerType, DeviceType
from msmodelslim.core.runner.layer_wise_runner import LayerWiseRunner
from msmodelslim.core.runner.pipeline_interface import PipelineInterface
from msmodelslim.core.runner.pipeline_parallel_runner import PPRunner
from msmodelslim.model import IModel
from msmodelslim.utils.exception import SchemaValidateError, UnsupportedError
from msmodelslim.utils.logging import get_logger, logger_setter
from msmodelslim.utils.seed import seed_all
from .quant_config import ModelslimV1QuantConfig
from ..interface import BaseQuantConfig


@logger_setter(prefix='msmodelslim.app.quant_service.modelslim_v1')
class ModelslimV1QuantService(BaseQuantService):
    backend_name: str = "modelslim_v1"

    def __init__(self, dataset_loader: DatasetLoaderInfra):
        super().__init__(dataset_loader)

    @staticmethod
    def _choose_runner_type(quant_config: ModelslimV1QuantConfig,
                            model_adapter: PipelineInterface,
                            device_indices: Optional[List[int]] = None) -> Literal[
        RunnerType.MODEL_WISE, RunnerType.LAYER_WISE]:
        """根据模型和配置确定使用的pipeline类型。

        Args:
            quant_config: 量化配置
            model_adapter: 模型适配器

        Returns:
            Literal['model_wise', 'layer_wise']: 确定的pipeline类型
        """
        if quant_config.spec.runner == RunnerType.MODEL_WISE:
            get_logger().info("Model-wise runner detected, using model-wise pipeline.")
            return RunnerType.MODEL_WISE

        if quant_config.spec.runner == RunnerType.LAYER_WISE:
            get_logger().info("Layer-wise runner detected, using layer-wise pipeline.")
            return RunnerType.LAYER_WISE

        if quant_config.spec.runner == RunnerType.DP_LAYER_WISE:
            get_logger().info("Distributed layer-wise runner detected, using distributed layer-wise pipeline.")
            return RunnerType.DP_LAYER_WISE

        if quant_config.spec.runner == RunnerType.AUTO and device_indices is not None and len(device_indices) > 1:
            get_logger().info("multi device configuration detected, using distributed layer-wise pipeline.")
            return RunnerType.DP_LAYER_WISE

        get_logger().info("Runner type not detected, using layer-wise pipeline.")
        return RunnerType.LAYER_WISE

    def quantize(
            self,
            quant_config: BaseQuantConfig,
            model_adapter: IModel,
            save_path: Optional[Path] = None,
            device: DeviceType = DeviceType.NPU,
            device_indices: Optional[List[int]] = None
    ):
        if not isinstance(quant_config, BaseQuantConfig):
            raise SchemaValidateError("task is NOT BaseQuantConfig",
                                      action="Please make sure the task is a BaseQuantConfig")
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
            ModelslimV1QuantConfig.from_base(quant_config),
            model_adapter, save_path, device, device_indices
        )

    def quant_process(self,
                      quant_config: ModelslimV1QuantConfig,
                      model_adapter: PipelineInterface,
                      save_path: Optional[Path],
                      device: DeviceType = DeviceType.NPU,
                      device_indices: Optional[List[int]] = None,
                      ):
        # clear quant_model_path before quantization
        if save_path and save_path.exists():
            # 只清除目录内容，不删除目录本身
            for item in save_path.iterdir():
                if item.is_dir():
                    shutil.rmtree(item)
                else:
                    item.unlink()
            get_logger().info("Cleared save_path: %s", save_path)

        common_seed = 42
        seed_all(seed=common_seed, mode=True)

        if device == DeviceType.NPU:
            # 如果使用npu进行量化需开启二进制编译，避免在线编译算子
            torch.npu.set_compile_mode(jit_compile=False)

        get_logger().info(f"==========QUANTIZATION: Prepare Dataset==========")
        dataset = self.dataset_loader.get_dataset_by_name(quant_config.spec.dataset)
        get_logger().info(f"prepare dataset from {quant_config.spec.dataset} success")

        final_process_cfg = quant_config.spec.process
        if save_path is not None:
            get_logger().info(f"==========QUANTIZATION: Prepare Persistence==========")
            for save_cfg in quant_config.spec.save:
                save_cfg.set_save_directory(save_path)

            # 注册处理器
            final_process_cfg += quant_config.spec.save
            get_logger().info(f"prepare Persistence to {save_path} success")

        get_logger().info(f"==========QUANTIZATION: Run Quantization==========")
        # 选择 runner
        runner_type = self._choose_runner_type(quant_config, model_adapter, device_indices)
        if runner_type == RunnerType.MODEL_WISE:
            runner = PPRunner(adapter=model_adapter)
        elif runner_type == RunnerType.LAYER_WISE:
            runner = LayerWiseRunner(adapter=model_adapter)
        elif runner_type == RunnerType.DP_LAYER_WISE:
            # 延迟导入以避免循环依赖
            from msmodelslim.core.runner.dp_layer_wise_runner import DPLayerWiseRunner
            runner = DPLayerWiseRunner(adapter=model_adapter)
        else:
            raise UnsupportedError("Invalid runner type",
                                   action="Please use RunnerType.MODEL_WISE or RunnerType.LAYER_WISE")

        get_logger().info(f"Create runner {runner_type} success")

        for process_cfg in final_process_cfg:
            runner.add_processor(processor_cfg=process_cfg)

        runner.run(calib_data=dataset, device=device, device_indices=device_indices)
        get_logger().info(f"==========QUANTIZATION: END==========")
