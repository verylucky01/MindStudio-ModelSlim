#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""
-------------------------------------------------------------------------
This file is part of the MindStudio project.
Copyright (c) 2025 Huawei Technologies Co.,Ltd.

MindStudio is licensed under Mulan PSL v2.
You can use this software according to the terms and conditions of the Mulan PSL v2.
You may obtain a copy of Mulan PSL v2 at:

         http://license.coscl.org.cn/MulanPSL2

THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
See the Mulan PSL v2 for more details.
-------------------------------------------------------------------------
"""
import argparse

import msmodelslim # do NOT remove, to trigger the patches
from msmodelslim.app.analysis.application import AnalysisMetrics
from msmodelslim.core.const import DeviceType, QuantType
from msmodelslim.utils.config import msmodelslim_config
from msmodelslim.utils.logging import set_logger_level
from msmodelslim.utils.validation.conversion import convert_to_bool

FAQ_HOME = "gitcode repo: Ascend/msmodelslim, wiki"
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
                                   "Note: Format 'device_type:index1,index2,...' is only supported "
                                   "when apiversion is 'modelslim_v1'. "
                                   "Default: 'npu' (single device)")
    quant_parser.add_argument('--config_path', type=str,
                              help="Explicit path to quantization config file")
    quant_parser.add_argument('--quant_type', type=QuantType, choices=QuantType,
                              help="Type of quantization to apply")
    quant_parser.add_argument('--trust_remote_code', type=convert_to_bool, default=False,
                              help="Trust custom code (bool type, must be True or False). "
                                   "Please ensure the security of the loaded custom code file.")
    quant_parser.add_argument("--debug", action="store_true",
                              help="Enable debug mode for context recording")
    quant_parser.add_argument('--tag', nargs='*', default=None,
                              help="Optional tag to match configs with verified scenario tags (e.g. mindie Atlas_A2_Inference, vllm cpu). "
                                   "User can add multiple tags; matching requires all tags to appear in the same scenario."
                                   "If user specifies this parameter but does not provide a hardware type tag, the current device type will be matched automatically.")

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
                                 help='Analysis metrics to use: std, quantile, kurtosis, attention_mse, mse_model_wise (default: kurtosis)')
    analysis_parser.add_argument('--calib_dataset', type=str, default='boolq.jsonl',
                                 help='Calibration dataset file path or filename in lab_calib directory. '
                                      'Supports .json and .jsonl formats (default: boolq.jsonl)')
    analysis_parser.add_argument('--topk', type=int, default=15,
                                 help='Number of top layers to output for disable_names '
                                      '(default: 15, empirical value, for reference only)')
    analysis_parser.add_argument('--trust_remote_code', type=convert_to_bool, default=False,
                                 help="Trust custom code (bool type, must be True or False). "
                                      "Please ensure the security of the loaded custom code file.")

    # auto tuning command
    tuning_parser = subparsers.add_parser('tune', help='Model quantization auto tuning tool')
    tuning_parser.add_argument('--model_type', type=str, default='default',
                                 help="Type of model to quantize (e.g. 'Qwen2.5-7B-Instruct', 'Qwen-QwQ-32B')")
    tuning_parser.add_argument('--model_path', required=True, type=str,
                                 help="Path to the original model")
    tuning_parser.add_argument('--save_path', required=True, type=str,
                              help="Path to save tuning results")
    tuning_parser.add_argument('--config', required=True, type=str,
                              help="Path to tuning config file")
    tuning_parser.add_argument('--device', type=str, default='npu',
                              help="Target device specification for quantization. "
                                   "Format: 'device_type' or 'device_type:index1,index2,...' "
                                   "(e.g., 'npu', 'npu:0,1,2,3', 'cpu'). "
                                   "Note: Format 'device_type:index1,index2,...' is only supported "
                                   "when apiversion is 'modelslim_v1'. "
                                   "Default: 'npu' (single device)")
    tuning_parser.add_argument('--timeout', type=str, default=None,
                               help='Timeout for tuning, e.g. 1D, 2H, 3D4H')
    tuning_parser.add_argument('--trust_remote_code', type=convert_to_bool, default=False,
                                 help="Trust custom code (bool type, must be True or False). "
                                      "Please ensure the security of the loaded custom code file.")

    args = parser.parse_args()
    if args.command == 'quant':
        from msmodelslim.cli.naive_quantization.__main__ import main as quant_main
        quant_main(args)
    elif args.command == 'analyze':
        from msmodelslim.cli.analysis.__main__ import main as analysis_main
        analysis_main(args)
    elif args.command == 'tune':
        from msmodelslim.cli.auto_tuning.__main__ import main as tuning_main
        tuning_main(args)
    else:
        # 可扩展其他组件
        parser.print_help()


if __name__ == '__main__':
    main()
