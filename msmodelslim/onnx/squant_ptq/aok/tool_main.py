# -*- coding: utf-8 -*-
# Copyright Huawei Technologies Co., Ltd. 2023-2023. All rights reserved.
import os
import argparse
from onnx import shape_inference
from msmodelslim import logger as msmodelslim_logger
from msmodelslim.onnx.squant_ptq.aok.utils.runner import Runner, create_runner
from msmodelslim.onnx.squant_ptq.aok.utils.utilities import load_model, define_batch_size, is_model_quantized, \
    check_and_fix_topology_sorting
from msmodelslim.onnx.squant_ptq.aok.optimizer.graph_optimizer import GraphOptimizer
from msmodelslim.onnx.squant_ptq.aok.optimizer.architectures import supported_architectures, AbstractArchitecture

IR_VERSION = 6
DEFAULT_BATCH_SIZE = 1


def optimize_model(model_path, args: argparse.Namespace,
                   opt_filter: int,
                   runner: Runner,
                   aok_model_path: str):
    optimizer = GraphOptimizer(msmodelslim_logger) \
        .set_opset_version(args.opset_version) \
        .set_ir_version(IR_VERSION) \
        .set_soc_version(args.soc_version) \
        .set_check_model(args.check_model) \
        .set_check_output_threshold(args.check_output_threshold) \
        .set_architecture(args.arch) \
        .set_debug(args.debug) \
        .set_runner(runner)

    baseline = optimizer.collect_inference_run_info(model_path)
    if args.arch in supported_architectures:
        opt_list = optimizer.find_optimizations(opt_filter)
        _, model_path_opt = optimizer.apply_all_optimizations(
            model_path, opt_list, debug=args.debug, aok_model_path=aok_model_path
        )
    else:
        if args.arch:
            msmodelslim_logger.info('Architecture not found')
            msmodelslim_logger.info('Start searching all optimizations')
        opt_list, _, _ = optimizer.search_optimizations(
            model_path, opt_filter, shut_down_structures=args.shut_down_structures, baseline=baseline
        )
        _, model_path_opt = optimizer.apply_all_optimizations(
            model_path, opt_list, debug=args.debug, aok_model_path=aok_model_path
        )

    opt_model = optimizer.collect_inference_run_info(model_path_opt)
    msmodelslim_logger.info(f'Finished: best model is {opt_model.model_name}')
    msmodelslim_logger.info(f'Baseline model inference time is {baseline.latency} ms.')
    msmodelslim_logger.info(f'Optimized model inference time is {opt_model.latency} ms.')
    return model_path_opt


def aok_export(model_path, cfg, aok_model_path=None):
    """
    Main function of ASCEND Optimizer Kit
    """
    logger = msmodelslim_logger

    folder_path, model_name_ext = os.path.split(model_path)
    model_name = os.path.splitext(model_name_ext)[0]

    model = load_model(folder_path, model_name, logger)
    if cfg.opset_version is None:
        cfg.opset_version = model.opset_import[0].version
    model = check_and_fix_topology_sorting(model, logger)
    model = shape_inference.infer_shapes(model)
    batch_size = define_batch_size(model, DEFAULT_BATCH_SIZE)
    opt_filter = AbstractArchitecture.OPT_FILTER_MASK_QUANT if is_model_quantized(model) \
        else AbstractArchitecture.OPT_FILTER_MASK_FP

    runner = create_runner(cfg, batch_size, logger)
    model_path_opt = optimize_model(model_path, cfg, opt_filter, runner, aok_model_path)
    return model_path_opt
