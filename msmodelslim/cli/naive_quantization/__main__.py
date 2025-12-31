# Copyright Huawei Technologies Co., Ltd. 2025. All rights reserved.
import os
from pathlib import Path

from msmodelslim.app.naive_quantization import NaiveQuantizationApplication
from msmodelslim.core.quant_service.proxy import QuantServiceProxy
from msmodelslim.cli.utils import parse_device_string
from msmodelslim.infra.file_dataset_loader import FileDatasetLoader
from msmodelslim.infra.vlm_dataset_loader import VLMDatasetLoader
from msmodelslim.infra.yaml_practice_manager import YamlPracticeManager
from msmodelslim.model import PluginModelFactory
from msmodelslim.utils.config import msmodelslim_config
from msmodelslim.utils.security.path import get_valid_read_path


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


def main(args):
    config_dir = get_practice_dir()
    custom_practice_dir = msmodelslim_config.env_vars.custom_practice_repo
    custom_practice_path = Path(custom_practice_dir) if custom_practice_dir else None
    practice_manager = YamlPracticeManager(
        official_config_dir=config_dir,
        custom_config_dir=custom_practice_path
    )
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
