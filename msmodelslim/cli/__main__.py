# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
import argparse

import msmodelslim # do NOT remove, to trigger the patches
from msmodelslim.app.analysis.application import AnalysisMetrics
from msmodelslim.core.const import DeviceType, QuantType
from msmodelslim.utils.config import msmodelslim_config
from msmodelslim.utils.logging import set_logger_level
from msmodelslim.utils.validation.conversion import convert_to_bool

FAQ_HOME = "gitcode repo: Ascend/msit/msmodelslim, wiki"
MIND_STUDIO_LOGO = "[Powered by MindStudio]"


def main():
    set_logger_level(msmodelslim_config.env_vars.log_level)

    parser = argparse.ArgumentParser(prog='msmodelslim',
                                     formatter_class=argparse.RawDescriptionHelpFormatter,
                                     description=f"MsModelSlim(MindStudio Model-Quantization Tools), "
                                                 f"{MIND_STUDIO_LOGO}.\n"
                                                 "Providing functions such as model quantization and compression "
                                                 "based on Ascend.\n"
                                                 f"For any issue, refer FAQ first: {FAQ_HOME}")
    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    # Quant command
    quant_parser = subparsers.add_parser('quant', help='Model quantization')
    quant_parser.add_argument('--model_type', required=True,
                              help="Type of model to quantize (e.g. 'Qwen2.5-7B-Instruct', 'Qwen-QwQ-32B')")
    quant_parser.add_argument('--model_path', required=True, type=str,
                              help="Path to the original model")
    quant_parser.add_argument('--save_path', required=True, type=str,
                              help="Path to save quantized model")
    quant_parser.add_argument('--device', type=str, default='npu',
                              help="Target device specification for quantization. "
                                   "Format: 'device_type' or 'device_type:index1,index2,...' "
                                   "(e.g., 'npu', 'npu:0,1,2,3', 'cpu'). "
                                   "Default: 'npu' (single device)")
    quant_parser.add_argument('--config_path', type=str,
                              help="Explicit path to quantization config file")
    quant_parser.add_argument('--quant_type', type=QuantType, choices=QuantType,
                              help="Type of quantization to apply")
    quant_parser.add_argument('--trust_remote_code', type=convert_to_bool, default=False,
                              help="Trust custom code (bool type, must be True or False). "
                                   "Please ensure the security of the loaded custom code file.")

    # Analyze command
    analysis_parser = subparsers.add_parser('analyze', help='Model quantization sensitivity analyze tool')
    analysis_parser.add_argument('--model_type', required=True,
                                 help="Type of model to quantize (e.g. 'Qwen2.5-7B-Instruct', 'Qwen-QwQ-32B')")
    analysis_parser.add_argument('--model_path', required=True, type=str,
                                 help="Path to the original model")
    analysis_parser.add_argument('--device', type=DeviceType, default=DeviceType.NPU, choices=DeviceType,
                                 help="Target device type for Analysis")
    analysis_parser.add_argument('--pattern',
                                 nargs='*',
                                 default=['*'],
                                 help='Pattern list to analyze (default is ["*"], means all match)')
    analysis_parser.add_argument('--metrics',
                                 type=AnalysisMetrics,
                                 default=AnalysisMetrics.KURTOSIS,
                                 choices=AnalysisMetrics,
                                 help='Analysis metrics to use: std, quantile, kurtosis (default: kurtosis)')
    analysis_parser.add_argument('--calib_dataset', type=str, default='boolq.jsonl',
                                 help='Calibration dataset file path or filename in lab_calib directory. '
                                      'Supports .json and .jsonl formats (default: boolq.jsonl)')
    analysis_parser.add_argument('--topk', type=int, default=15,
                                 help='Number of top layers to output for disable_names '
                                      '(default: 15, empirical value, for reference only)')
    analysis_parser.add_argument('--trust_remote_code', type=convert_to_bool, default=False,
                                 help="Trust custom code (bool type, must be True or False). "
                                      "Please ensure the security of the loaded custom code file.")
    args = parser.parse_args()

    if args.command == 'quant':
        from msmodelslim.cli.naive_quantization.__main__ import main as quant_main
        quant_main(args)
    elif args.command == 'analyze':
        from msmodelslim.cli.analysis.__main__ import main as analysis_main
        analysis_main(args)
    else:
        # 可扩展其他组件
        parser.print_help()


if __name__ == '__main__':
    main()
