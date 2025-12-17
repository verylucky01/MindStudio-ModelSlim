# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
from pathlib import Path
from typing import List, Any, Generator

import torch.nn as nn

from msmodelslim.core.base.protocol import ProcessRequest
from msmodelslim.core.const import DeviceType
from msmodelslim.model.common.layer_wise_forward import transformers_generated_forward_func, \
    generated_decoder_layer_visit_func
from msmodelslim.utils.exception import InvalidModelError
from msmodelslim.utils.exception_decorator import exception_handler
from msmodelslim.utils.logging import logger_setter, get_logger
from ..common.transformers import TransformersModel
from ..interface_hub import ModelInfoInterface, ModelSlimPipelineInterfaceV0, ModelSlimPipelineInterfaceV1, \
    AnalyzePipelineInterface


@logger_setter()
class DefaultModelAdapter(TransformersModel,
                          ModelInfoInterface,  # support naive quantization
                          ModelSlimPipelineInterfaceV0,  # support modelslim v0
                          ModelSlimPipelineInterfaceV1,  # support modelslim v1
                          AnalyzePipelineInterface,  # support analyse
                          ):
    """
    Default model adapter which implements some widely used interface.
    You can try to quant new unadapted model by using this model adapter.
    HOWEVER, it may be not functional.
    You can treat this model adapter as a reference to implement your own model adapter.
    """

    def __init__(self,
                 model_type: str,
                 model_path: Path,
                 trust_remote_code: bool = False):
        get_logger().warning(f"You are using default model adapter, "
                             f"which may be not functional. "
                             f"Please implement register custom model adapter for your model.")
        with exception_handler('You are creating default model adapter but failed',
                               ms_err_cls=InvalidModelError,
                               action='Please ensure default model adapter match your model'):
            super().__init__(model_type, model_path, trust_remote_code)

    def get_model_type(self) -> str:
        get_logger().warning(f"You are using default get_model_type, "
                             f"which may be not functional. "
                             f"Please implement get_model_type for your model.")
        return self.model_type

    def get_model_pedigree(self) -> str:
        get_logger().warning(f"You are using default get_model_pedigree, "
                             f"which may be not functional. "
                             f"Please implement get_model_pedigree for your model.")
        return self.model_pedigree

    def load_model(self, device: DeviceType = DeviceType.NPU) -> nn.Module:
        get_logger().warning(f"You are using default load_model, "
                             f"which may be not functional. "
                             f"Please implement load_model for your model.")
        with exception_handler('You are using default load_model to load model but failed',
                               ms_err_cls=InvalidModelError,
                               action='Please ensure default model adapter match your model'):
            return self._load_model(device)

    def handle_dataset(self, dataset: Any, device: DeviceType = DeviceType.NPU) -> List[Any]:
        get_logger().warning(f"You are using default generate_dataset, "
                             f"which may be not functional. "
                             f"Please implement generate_dataset for your model.")
        with exception_handler('You are using default handle_dataset but failed',
                               ms_err_cls=InvalidModelError,
                               action='Please ensure default model adapter match your model'):
            return self._get_tokenized_data(dataset, device)

    def handle_dataset_by_batch(self,
                                dataset: Any,
                                batch_size: int,
                                device: DeviceType = DeviceType.NPU) -> List[Any]:
        get_logger().warning(f"You are using default generate_dataset_by_batch, "
                             f"which may be not functional. "
                             f"Please implement generate_dataset_by_batch for your model.")
        with exception_handler('You are using default handle_dataset_by_batch but failed',
                               ms_err_cls=InvalidModelError,
                               action='Please ensure default model adapter match your model'):
            return self._get_batch_tokenized_data(calib_list=dataset,
                                                  batch_size=batch_size,
                                                  device=device)

    def init_model(self, device: DeviceType = DeviceType.NPU) -> nn.Module:
        get_logger().warning(f"You are using default load_model, "
                             f"which may be not functional. "
                             f"Please implement load_model for your model.")
        with exception_handler('You are using default init_model to init model but failed',
                               ms_err_cls=InvalidModelError,
                               action='Please ensure default model adapter match your model'):
            return self._load_model(device)

    def generate_model_visit(self, model: nn.Module) -> Generator[ProcessRequest, Any, None]:
        get_logger().warning(f"You are using default generate_model_visit, "
                             f"which may be not functional. "
                             f"Please implement generate_model_visit for your model.")
        with exception_handler('You are using default generate_model_visit but failed',
                               ms_err_cls=InvalidModelError,
                               action='Please ensure default model adapter match your model'):
            yield from generated_decoder_layer_visit_func(model)

    def generate_model_forward(self, model: nn.Module, inputs: Any,
                               ) -> Generator[ProcessRequest, Any, None]:
        get_logger().warning(f"You are using default generate_model_forward, "
                             f"which may be not functional. "
                             f"Please implement generate_model_forward for your model.")
        with exception_handler('You are using default generate_model_forward but failed',
                               ms_err_cls=InvalidModelError,
                               action='Please ensure default model adapter match your model'):
            yield from transformers_generated_forward_func(model, inputs)

    def enable_kv_cache(self, model: nn.Module, need_kv_cache: bool) -> None:
        get_logger().warning(f"You are using default enable_kv_cache, "
                             f"which may be not functional. "
                             f"Please implement enable_kv_cache for your model.")
        with exception_handler('You are using default enable_kv_cache but failed',
                               ms_err_cls=InvalidModelError,
                               action='Please ensure default model adapter match your model'):
            return self._enable_kv_cache(model, need_kv_cache)
