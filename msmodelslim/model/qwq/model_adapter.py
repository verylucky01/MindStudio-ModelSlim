# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
from typing import Any, List, Generator

from torch import nn

from msmodelslim.core.base.protocol import ProcessRequest
from msmodelslim.core.const import DeviceType
from msmodelslim.utils.logging import logger_setter
from ..common.layer_wise_forward import generated_decoder_layer_visit_func, transformers_generated_forward_func
from ..common.transformers import TransformersModel
from ..interface_hub import ModelInfoInterface, ModelSlimPipelineInterfaceV0, ModelSlimPipelineInterfaceV1, \
    AnalyzePipelineInterface


@logger_setter()
class QwqModelAdapter(TransformersModel,
                      ModelInfoInterface,
                      ModelSlimPipelineInterfaceV0,
                      ModelSlimPipelineInterfaceV1,
                      AnalyzePipelineInterface,
                      ):
    def get_model_type(self) -> str:
        return self.model_type

    def get_model_pedigree(self) -> str:
        return 'qwq'

    def load_model(self, device: DeviceType = DeviceType.NPU) -> nn.Module:
        return self._load_model(device)

    def handle_dataset(self, dataset: Any, device: DeviceType = DeviceType.NPU) -> List[Any]:
        return self._get_tokenized_data(dataset, device)

    def handle_dataset_by_batch(self,
                                dataset: Any,
                                batch_size: int,
                                device: DeviceType = DeviceType.NPU) -> List[Any]:
        return self._get_batch_tokenized_data(calib_list=dataset,
                                              batch_size=batch_size,
                                              device=device)

    def init_model(self, device: DeviceType = DeviceType.NPU) -> nn.Module:
        return self._load_model(device)

    def generate_model_visit(self, model: nn.Module) -> Generator[ProcessRequest, Any, None]:
        yield from generated_decoder_layer_visit_func(model)

    def generate_model_forward(self, model: nn.Module, inputs: Any,
                               ) -> Generator[ProcessRequest, Any, None]:
        yield from transformers_generated_forward_func(model, inputs)

    def enable_kv_cache(self, model: nn.Module, need_kv_cache: bool) -> None:
        return self._enable_kv_cache(model, need_kv_cache)
