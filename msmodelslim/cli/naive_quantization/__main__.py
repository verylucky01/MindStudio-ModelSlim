# Copyright Huawei Technologies Co., Ltd. 2025. All rights reserved.
import os
from pathlib import Path
from typing import Tuple, Optional, List

from msmodelslim.core.const import DeviceType
from msmodelslim.app.naive_quantization import NaiveQuantizationApplication
from msmodelslim.app.quant_service.proxy import QuantServiceProxy
from msmodelslim.infra.dataset_loader import FileDatasetLoader
from msmodelslim.infra.vlm_dataset_loader import VLMDatasetLoader
from msmodelslim.infra.practice_manager import PracticeManager
from msmodelslim.model import PluginModelFactory
from msmodelslim.utils.security.path import get_valid_read_path
from msmodelslim.utils.exception import SchemaValidateError


def get_practice_dir():
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    lab_practice_dir = os.path.abspath(os.path.join(cur_dir, '../../lab_practice'))
    lab_practice_dir = get_valid_read_path(lab_practice_dir, is_dir=True)
    return Path(lab_practice_dir)


def get_dataset_dir():
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    lab_calib_dir = os.path.abspath(os.path.join(cur_dir, '../../lab_calib'))
    lab_calib_dir = get_valid_read_path(lab_calib_dir, is_dir=True)
    return Path(lab_calib_dir)


def parse_device_string(device_str: str) -> Tuple[str, Optional[List[int]]]:
    """
    Parse device string into device type string and device indices list.
    
    Args:
        device_str: Device specification string (e.g., 'npu', 'npu:0,1,2,3', 'cpu')
    
    Returns:
        device_type: str, device type string
        device_indices: Optional[List[int]], list of device indices, None if not specified
    
    Raises:
        SchemaValidateError: If device_str is empty or device indices cannot be converted to integers
    
    Examples:
        >>> parse_device_string('npu')
        ('npu', None)
        >>> parse_device_string('npu:0,1,2,3')
        ('npu', [0, 1, 2, 3])
        >>> parse_device_string('cpu')
        ('cpu', None)
    """
    device_str = device_str.strip()
    if not device_str:
        raise SchemaValidateError("device string cannot be empty")
    
    # Split by colon to separate device type and indices
    parts = device_str.split(':', 1)
    device_type_str = parts[0].strip()
    
    # Validate and convert device type
    try:
        device_type = DeviceType(device_type_str)
    except ValueError as e:
        valid_types = ', '.join([f"'{dt.value}'" for dt in DeviceType])
        raise SchemaValidateError(
            f"Invalid device type: '{device_type_str}'. "
            f"Supported device types: {valid_types}"
        ) from e
    
    # Parse device indices if provided
    device_indices = None
    if len(parts) > 1:
        indices_str = parts[1].strip()
        if indices_str:  # Only process if not empty
            # Split by comma and convert to integers
            indices_list = [idx.strip() for idx in indices_str.split(',') if idx.strip()]
            
            if not indices_list:
                raise SchemaValidateError(
                    f"Device indices cannot be empty after parsing: '{indices_str}'"
                )
            
            # Convert to integers
            try:
                device_indices = [int(idx) for idx in indices_list]
            except ValueError as e:
                raise SchemaValidateError(
                    f"Invalid device indices format: '{indices_str}'. "
                    f"Expected comma-separated integers (e.g., '0,1,2,3')"
                ) from e
    
    return device_type, device_indices


def main(args):
    config_dir = get_practice_dir()
    practice_manager = PracticeManager(official_config_dir=config_dir)
    dataset_dir = get_dataset_dir()
    dataset_loader = FileDatasetLoader(dataset_dir)
    vlm_dataset_loader = VLMDatasetLoader(dataset_dir)

    device_type, device_index = parse_device_string(args.device)
    
    quant_service = QuantServiceProxy(dataset_loader, vlm_dataset_loader)
    
    app = NaiveQuantizationApplication(
        practice_manager=practice_manager,
        quant_service=quant_service,
        model_factory=PluginModelFactory(),
    )

    app.quant(
        model_type=args.model_type,
        model_path=args.model_path,
        save_path=args.save_path,
        device_type=device_type,
        device_index=device_index,
        quant_type=args.quant_type,
        config_path=args.config_path,
        trust_remote_code=args.trust_remote_code
    )