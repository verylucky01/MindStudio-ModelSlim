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
import functools
import os
import copy
from pathlib import Path
from typing import Optional, List
from msmodelslim.core.quant_service.base import BaseQuantService
from msmodelslim.core.quant_service.dataset_loader_infra import DatasetLoaderInfra
from msmodelslim.core.const import DeviceType
from msmodelslim.core.runner.layer_wise_runner import LayerWiseRunner
from msmodelslim.model import IModel
from msmodelslim.utils.cache import load_cached_data_for_models, to_device
from msmodelslim.utils.exception import SchemaValidateError
from msmodelslim.utils.logging import get_logger, logger_setter
from .pipeline_interface import MultimodalPipelineInterface
from .quant_config import MultimodalSDModelslimV1QuantConfig, MultiExpertQuantConfig
from ..interface import BaseQuantConfig



@logger_setter(prefix='msmodelslim.core.quant_service.multimodal_sd_v1')  # 4-level: msmodelslim.core.quant_service.multimodal_sd_v1
class MultimodalSDModelslimV1QuantService(BaseQuantService):
    backend_name: str = "multimodal_sd_modelslim_v1"

    def __init__(self, dataset_loader: DatasetLoaderInfra):
        super().__init__(dataset_loader)

    def quantize(
            self,
            quant_config: BaseQuantConfig,
            model_adapter: IModel,
            save_path: Optional[Path] = None,
            device: DeviceType = DeviceType.NPU,
            device_indices: Optional[List[int]] = None,
    ) -> None:
        if not isinstance(quant_config, BaseQuantConfig):
            raise SchemaValidateError("task must be a BaseTask")
        if not isinstance(model_adapter, MultimodalPipelineInterface):
            raise SchemaValidateError("model must be a MultimodalPipelineInterface")
        if save_path is not None and not isinstance(save_path, Path):
            raise SchemaValidateError("save_path must be a Path or None")
        if not isinstance(device, DeviceType):
            raise SchemaValidateError("device must be a DeviceType")

        return self.quant_process(MultimodalSDModelslimV1QuantConfig.from_base(quant_config), model_adapter, save_path,
                                  device)

    def quantize_multi_expert_models(self, config: MultiExpertQuantConfig):
        model_adapter = config.model_adapter
        models = config.models
        calib_data = config.calib_data
        quant_config = config.quant_config
        save_path = config.save_path
        device = config.device

        # 保存原始transformer以备恢复
        original_transformer = model_adapter.transformer

        # 遍历所有专家模型进行量化
        for expert_name, expert_model in models.items():
            get_logger().info(f"========== Quantizing {model_adapter.model_args.task_config}_{expert_name} ==========")

            if expert_name not in calib_data:
                get_logger().error(f"========== \
                    Calib data missing {model_adapter.model_args.task_config}_{expert_name}, continued ==========")
                continue

            model_adapter.transformer = expert_model

            # 自动生成专家专属保存路径
            if expert_name != '':
                expert_save_path = save_path.joinpath(f"{model_adapter.model_args.task_config}_{expert_name}")
                expert_save_path.mkdir(parents=True, exist_ok=True)
            else:
                expert_save_path = save_path

            final_process_cfg = copy.copy(quant_config.spec.process)

            if expert_save_path is not None:
                get_logger().warning(f"========== QUANTIZATION: Prepare Save Path ==========")
                for save_cfg in quant_config.spec.save:
                    save_cfg.set_save_directory(expert_save_path)
                final_process_cfg += quant_config.spec.save
                get_logger().warning(f"prepare Persistence to {expert_save_path} success")

            if quant_config.spec.runner != "layer_wise":
                get_logger().warning(f"runner for multimodal_sd_v1 is not layer_wise, will be converted to layer_wise.")

            runner = LayerWiseRunner(adapter=model_adapter)
            for process_cfg in final_process_cfg:
                runner.add_processor(processor_cfg=process_cfg)

            try:
                model_adapter.apply_quantization(functools.partial(runner.run,
                    calib_data=calib_data[expert_name], device=device, model=expert_model))
                get_logger().info(f"========== {expert_name} quantized, save to {expert_save_path} ==========")
            except Exception as e:
                get_logger().error(f"========== {expert_name} quantization failed: {str(e)} ==========")
                raise RuntimeError(f"========== {expert_name} quantization failed: {str(e)} ==========") from e

        model_adapter.transformer = original_transformer

    def quant_process(self, quant_config: MultimodalSDModelslimV1QuantConfig,
                  model_adapter: MultimodalPipelineInterface,
                  save_path: Optional[Path], device: DeviceType = DeviceType.NPU):

        model_adapter.set_model_args(quant_config.spec.multimodal_sd_config.model_extra['model_config'])
        model_adapter.load_pipeline()

        get_logger().info(f"==========QUANTIZATION: Prepare Dataset==========")

        models = model_adapter.init_model(device)

        config_dump_data_dir = quant_config.spec.multimodal_sd_config.dump_config.dump_data_dir
        if config_dump_data_dir:
            base_dir = config_dump_data_dir
        else:
            base_dir = save_path

        pth_file_path_list = {}
        for expert_name, _ in models.items():
            pth_file_path_list[expert_name] = os.path.join(base_dir,
                f"calib_data_{model_adapter.model_args.task_config}_{expert_name}.pth")

        calib_data = load_cached_data_for_models(
            pth_file_path_list=pth_file_path_list,
            generate_func=model_adapter.run_calib_inference,
            models=models,
            dump_config=quant_config.spec.multimodal_sd_config.dump_config
        )

        get_logger().info(f"prepare calib_data from {base_dir} success")

        calib_data = to_device(calib_data, device.value)
        get_logger().info(f"==========QUANTIZATION: Run Quantization==========")

        config = MultiExpertQuantConfig(
            model_adapter=model_adapter,
            models=models,
            calib_data=calib_data,
            quant_config=quant_config,
            save_path=save_path,
            device=device
        )

        self.quantize_multi_expert_models(config)

        get_logger().info(f"==========QUANTIZATION: END==========")
