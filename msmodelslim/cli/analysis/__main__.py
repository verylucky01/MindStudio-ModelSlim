# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
import os
from pathlib import Path

from msmodelslim.app.analysis import LayerAnalysisApplication
from msmodelslim.core.analysis_service import LayerSelectorAnalysisService
from msmodelslim.infra.file_dataset_loader import FileDatasetLoader
from msmodelslim.model import PluginModelFactory
from msmodelslim.utils.logging import get_logger
from msmodelslim.utils.security.path import get_valid_read_path


def get_dataset_dir():
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    lab_calib_dir = os.path.abspath(os.path.join(cur_dir, '../../lab_calib'))
    lab_calib_dir = get_valid_read_path(lab_calib_dir, is_dir=True)
    return Path(lab_calib_dir)


def main(args):
    """Main function for layer analysis CLI"""
    try:
        # Get dataset directory
        dataset_dir = get_dataset_dir()
        # Create dataset loader
        dataset_loader = FileDatasetLoader(dataset_dir)

        # Create analysis service
        analysis_service = LayerSelectorAnalysisService(dataset_loader)
        # Create model factory
        model_factory = PluginModelFactory()
        # Create analysis app
        analysis_app = LayerAnalysisApplication(
            analysis_service=analysis_service,
            model_factory=model_factory,
        )

        # Run analysis
        result = analysis_app.analyze(
            model_type=args.model_type,
            model_path=args.model_path,
            patterns=args.pattern,
            device=args.device,
            metrics=args.metrics,
            calib_dataset=args.calib_dataset,
            topk=args.topk,
            trust_remote_code=args.trust_remote_code
        )
        return result

    except Exception as e:
        get_logger().error(f"Layer analysis failed: {str(e)}")
        raise
