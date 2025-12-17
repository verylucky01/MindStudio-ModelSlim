# Copyright Huawei Technologies Co., Ltd. 2025. All rights reserved.
import re
from pathlib import Path
from typing import Callable, Optional, List, Tuple
import torch

from msmodelslim.core.const import DeviceType
from msmodelslim.core.const import QuantType
from msmodelslim.model import IModelFactory, IModel
from msmodelslim.utils.exception import SchemaValidateError, ToDoError, UnsupportedError
from msmodelslim.utils.exception_decorator import exception_catcher
from msmodelslim.utils.logging import logger_setter, get_logger
from msmodelslim.utils.security import yaml_safe_load
from msmodelslim.utils.validation.conversion import (
    convert_to_readable_file,
    convert_to_writable_dir,
    convert_to_readable_dir
)
from msmodelslim.utils.validation.value import validate_str_length
from .model_info_interface import ModelInfoInterface
from .practice_manager_infra import PracticeManagerInfra
from ..practice.interface import PracticeConfig
from ..quant_service import IQuantService

DEFAULT_PEDIGREE = 'default'
DEFAULT_CONFIG_ID = 'default'


def validate_device_index(device_index: Optional[List[int]], device_type: DeviceType):
    """
    Validate device_index parameter.

    Args:
        device_index: Device indices to validate
        device_type: Device type for context validation

    Raises:
        SchemaValidateError: If device_index is invalid
    """

    # Value validation: check if indices are non-negative
    if any(idx < 0 for idx in device_index):
        negative_indices = [idx for idx in device_index if idx < 0]
        raise SchemaValidateError(
            f"Device indices must be non-negative integers, "
            f"but got negative values: {negative_indices}"
        )

    # Value validation: check for duplicates
    if len(device_index) != len(set(device_index)):
        duplicates = [idx for idx in set(device_index) if device_index.count(idx) > 1]
        raise SchemaValidateError(
            f"Device indices must be unique, but found duplicates: {duplicates}"
        )

    # CPU does not support multi-device
    if device_type == DeviceType.CPU and len(device_index) > 1:
        raise SchemaValidateError(
            f"CPU does not support multi-device configuration. "
            f"Got device indices: {device_index}. "
            f"Please use NPU for multi-device parallel, or use single CPU device."
        )

    # Value validation: check device availability
    if device_type == DeviceType.NPU:
        max_device_count = torch.npu.device_count()
    else:
        # CPU doesn't need device count validation
        max_device_count = None

    # Check if indices exceed available devices
    if max_device_count is not None:
        invalid_indices = [idx for idx in device_index if idx >= max_device_count]
        if invalid_indices:
            raise SchemaValidateError(
                f"Device indices {invalid_indices} exceed maximum available device index "
                f"({max_device_count - 1}). Available device indices: 0 to {max_device_count - 1}"
            )


@logger_setter('msmodelslim.app.naive_quantization')
class NaiveQuantizationApplication:

    def __init__(
            self,
            practice_manager: PracticeManagerInfra,
            quant_service: IQuantService,
            model_factory: IModelFactory,
    ):
        self.practice_manager = practice_manager
        self.quant_service = quant_service
        self.model_factory = model_factory

    @staticmethod
    def check_label(label, w_bit, a_bit, use_kv_cache, is_sparse):
        """Check if the label matches the quantization parameters"""
        if label.get('w_bit') != w_bit:
            return False
        if label.get('a_bit') != a_bit:
            return False
        if is_sparse ^ label.get('is_sparse', False):
            return False
        if use_kv_cache ^ label.get('kv_cache', False):
            return False
        return True

    def get_default_practice(self,
                             prompt="No configuration found.",
                             error_msg="The corresponding configuration is not currently supported"
                             ) -> PracticeConfig:
        user_input = input(
            prompt +
            " Default configuration will be used. (Enter y to continue, otherwise it will exit): ").strip().lower()[:3]
        if user_input != 'y':
            raise UnsupportedError(error_msg,
                                   action="Please write your own configuration file and use via --config_path.")
        return self.practice_manager.get_config_by_id(DEFAULT_PEDIGREE, DEFAULT_CONFIG_ID)

    def get_best_practice(self,
                          model_adapter: IModel,
                          quant_type: Optional[QuantType] = None,
                          config_path: Optional[Path] = None,
                          ) -> PracticeConfig:

        # Handle explicit config path
        if config_path is not None:
            config_dict = yaml_safe_load(str(config_path))
            config = PracticeConfig.from_dict(config_dict)
            get_logger().info(f"Naive Quant apply config_path: {config_path}")
            return config

        if not isinstance(model_adapter, ModelInfoInterface):
            raise ToDoError(f"Model adapter {model_adapter.__class__.__name__} "
                            f"does NOT implement ModelInfoInterface",
                            action="Please implement ModelInfoInterface to support get best practice.")

        model_type = model_adapter.get_model_type()
        model_pedigree = model_adapter.get_model_pedigree()

        # Handle unknown model
        if model_pedigree not in self.practice_manager:
            return self.get_default_practice(
                prompt=f"No matching configuration found for model_pedigree={model_pedigree}.")

        # Handle quant_type matching
        if quant_type is None:
            raise ValueError(f"Quant_type must be provided")

        # Parse quant_type parameters
        match_result = re.match(r'^w(\d+)a(\d+)(c?8?)(s?)$', quant_type.value)
        if not match_result:
            raise ValueError(f"Invalid quant_type format: {quant_type.value}")
        w_bit = int(match_result.group(1))
        a_bit = int(match_result.group(2))
        use_kv_cache = bool(match_result.group(3))
        is_sparse = bool(match_result.group(4))

        for config in self.practice_manager.iter_config(model_pedigree):
            if config.metadata.verified_model_types and model_type not in config.metadata.verified_model_types:
                continue

            if not self.check_label(config.metadata.label, w_bit, a_bit, use_kv_cache, is_sparse):
                continue

            get_logger().info(f"Naive Quant apply config_id: {config.metadata.config_id}")
            return config

        return self.get_default_practice(prompt=f"No matching configuration found for model_type={model_type}.")

    @exception_catcher
    def quant(self,
              model_type: str,
              model_path: str,
              save_path: str,
              device_type: DeviceType = DeviceType.NPU,
              device_index: Optional[List[int]] = None,
              quant_type: Optional[QuantType] = None,
              config_path: Optional[str] = None,
              trust_remote_code: bool = False):
        """
        Run the naive quantization application.
        Args:
            model_type: str, the type of the model
            model_path: str, the path of the model
            save_path: str, the path to save the quantized model
            device_type: DeviceType, the type of device (e.g., DeviceType.NPU, DeviceType.CPU)
                        Default: DeviceType.NPU
            device_index: Optional[List[int]], list of device indices to use (e.g., [0, 1, 2, 3])
                         If None, uses single default device
                         Default: None
            quant_type: Optional[QuantType], the quantization type, config_path and quant_type only one can be provided
            config_path: Optional[str], the path to config file, config_path and quant_type only one can be provided
            trust_remote_code: bool, whether to trust the remote code
        """
        # 字符串类型与长度校验
        str_params = [
            ("model_type", model_type),
            ("model_path", model_path),
            ("save_path", save_path)
        ]
        for param_name, value in str_params:
            if not isinstance(value, str):
                raise SchemaValidateError(f"{param_name} must be a string, but got {type(value)}")
            validate_str_length(input_str=value, str_name=param_name)

        model_path = convert_to_readable_dir(model_path)
        if not isinstance(model_path, Path):
            raise SchemaValidateError(f"model_path must be a Path, but got {type(model_path)}")
        save_path = convert_to_writable_dir(save_path)
        if not isinstance(save_path, Path):
            raise SchemaValidateError(f"save_path must be a Path, but got {type(save_path)}")
        if not isinstance(device_type, DeviceType):
            raise SchemaValidateError(f"device_type must be a DeviceType, but got {type(device_type)}")
        if device_index is not None:
            validate_device_index(device_index, device_type)
        if config_path is not None:
            validate_str_length(input_str=config_path, str_name='config_path')
            config_path = convert_to_readable_file(config_path)
        if not ((quant_type is None) ^ (config_path is None)):
            raise SchemaValidateError(f"quant_type and config_path only one can be provided")
        if quant_type is not None and not isinstance(quant_type, QuantType):
            raise SchemaValidateError(f"quant_type must be a QuantType")
        if config_path is not None and not isinstance(config_path, Path):
            raise SchemaValidateError(f"config_path must be a Path, but got {type(config_path)}")
        if not isinstance(trust_remote_code, bool):
            raise SchemaValidateError(f"trust_remote_code must be a bool")

        # Log parameters
        get_logger().info(f'quantization with following parameters:')
        get_logger().info(f"model_type: {model_type}")
        get_logger().info(f"model_path: {model_path}")
        get_logger().info(f"save_path: {save_path}")
        get_logger().info(f"device_type: {device_type}")
        if device_index is not None and len(device_index) > 1:
            device_list = ','.join(map(str, device_index))
            get_logger().info(
                f"using {len(device_index)} devices: {device_type.value}:{device_list}"
            )
        elif device_index is not None and len(device_index) == 1:
            get_logger().info(f"using single device: {device_type.value}:{device_index[0]}")
        else:
            get_logger().info(f"using single device (default): {device_type.value}")
        if quant_type is not None:
            get_logger().info(f"quant_type: {quant_type}")
        if config_path is not None:
            get_logger().info(f"config_path: {config_path}")
        get_logger().info(f"trust_remote_code: {trust_remote_code}")

        self._quant(
            model_type, model_path, save_path, device_type,
            device_index, quant_type, config_path, trust_remote_code
        )

    def _quant(
            self,
            model_type: str,
            model_path: Path,
            save_path: Path,
            device_type: DeviceType = DeviceType.NPU,
            device_index: Optional[List[int]] = None,
            quant_type: Optional[QuantType] = None,
            config_path: Optional[Path] = None,
            trust_remote_code: bool = False
    ):
        get_logger().info(f"===========ANALYSE MODEL===========")
        model_adapter = self.model_factory.create(
            model_type, model_path, trust_remote_code
        )
        get_logger().info(f"Using model adapter {model_adapter.__class__.__name__}.")

        get_logger().info(f"===========GET BEST PRACTICE===========")
        practice_config = self.get_best_practice(
            model_adapter=model_adapter,
            quant_type=quant_type,
            config_path=config_path
        )
        get_logger().info(f"Get best practice {practice_config.metadata.config_id} success.")

        get_logger().info(f"===========QUANTIZE MODEL===========")
        self.quant_service.quantize(
            quant_config=practice_config.extract_quant_config(),
            model_adapter=model_adapter,
            save_path=save_path,
            device=device_type,
            device_indices=device_index
        )
        get_logger().info(f"===========SUCCESS===========")
