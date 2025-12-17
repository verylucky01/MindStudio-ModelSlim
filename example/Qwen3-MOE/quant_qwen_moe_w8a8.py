# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
import argparse
import functools
import json
import os
import sys

import torch
import torch_npu

current_directory = os.path.dirname(os.path.abspath(__file__))
parent_directory = os.path.abspath(os.path.join(current_directory, '..', ".."))
sys.path.append(parent_directory)

from example.common.security.path import get_valid_read_path, get_write_directory
from example.common.security.type import check_number
from example.common.utils import SafeGenerator, cmd_bool
from example.common.rot_utils.rot_qwen import rot_model
from msmodelslim.tools.copy_config_files import copy_config_files, modify_config_json
from msmodelslim.pytorch.llm_ptq.anti_outlier import AntiOutlierConfig, AntiOutlier
from msmodelslim.pytorch.llm_ptq.llm_ptq_tools import Calibrator, QuantConfig
from msmodelslim.utils.logging import set_logger_level
from msmodelslim import logger


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, help="The path of float model and tokenizer"),
    parser.add_argument('--save_path', type=str, help="The path to save quant model"),
    parser.add_argument('--layer_count', type=int, default=0, help="Layer count when loading model")
    parser.add_argument('--anti_dataset', type=str, default="../common/qwen3-moe_anti_prompt_50.json",
                        help="The calib data for anti outlier")
    parser.add_argument('--calib_dataset', type=str, default="../common/qwen3-moe_calib_prompt_50.json",
                        help="The calib data for calibration")
    parser.add_argument('--batch_size', type=int, default=4, help="Batch size for anti and calibration")
    parser.add_argument('--mindie_format', action="store_true", help="Enable only mindie config save")
    parser.add_argument('--trust_remote_code', type=cmd_bool, default=False)
    parser.add_argument('--rot', action='store_true', help="rot model")
    return parser.parse_args()


def custom_hook(model_config):
    model_config["quantize"] = "w8a8_dynamic"


def get_calib_dataset_batch(model_tokenizer, calib_list, batch_size, device="npu"):
    calib_dataset = []
    calib_list = [calib_list[i:i + batch_size] for i in range(0, len(calib_list), batch_size)]
    for calib_data in calib_list:
        inputs = model_tokenizer(calib_data, return_tensors='pt', padding=True).to(device)
        calib_dataset.append(
            [value.to(device) for key, value in inputs.data.items() if isinstance(value, torch.Tensor)])
    return calib_dataset


def main():
    args = parse_args()
    set_logger_level("info")

    model_path = args.model_path
    batch_size = args.batch_size

    save_path = get_write_directory(args.save_path, write_mode=0o750)
    check_number(batch_size, int, 1, 16, "batch_size")

    safe_generator = SafeGenerator()

    config = safe_generator.get_config_from_pretrained(model_path=model_path,
                                                       trust_remote_code=args.trust_remote_code)
    num_layer = config.num_hidden_layers
    if args.layer_count < 0 or args.layer_count > num_layer:
        raise ValueError(
            f"Invalid value for parameter layer_count: {args.layer_count}."
            f"Must be between 0 and {num_layer}."
        )
    # Set layer count to 0 means use all layers, otherwise it will only use the first layer_count layers
    config.num_hidden_layers = args.layer_count if args.layer_count != 0 else config.num_hidden_layers
    # Disable use cache because we don't need to use cache, otherwise it will use too much device memory then cause OOM
    config.use_cache = False

    tokenizer = safe_generator.get_tokenizer_from_pretrained(model_path=model_path,
                                                             config=config,
                                                             trust_remote_code=args.trust_remote_code,
                                                             use_fast=True,
                                                             add_eos_token=True)

    model = safe_generator.get_model_from_pretrained(model_path=model_path,
                                                     config=config,
                                                     trust_remote_code=args.trust_remote_code,
                                                     device_map={
                                                         "model.embed_tokens": 0,
                                                         "model.layers": "cpu",
                                                         "model.norm": "cpu",
                                                         "lm_head": 0,
                                                     },
                                                     torch_dtype="auto",
                                                     attn_implementation='eager')

    anti_dataset_path = get_valid_read_path(args.anti_dataset, "json", is_dir=False)
    calib_dataset_path = get_valid_read_path(args.calib_dataset, "json", is_dir=False)
    with open(anti_dataset_path, "r") as file:
        anti_prompt = json.load(file)
    with open(calib_dataset_path, "r") as file:
        calib_prompt = json.load(file)
    anti_dataset = get_calib_dataset_batch(tokenizer, anti_prompt, batch_size, model.device)
    dataset_calib = get_calib_dataset_batch(tokenizer, calib_prompt, batch_size, model.device)

    with torch.no_grad():

        test_prompt = "what is deep learning?"
        test_input = tokenizer(test_prompt, return_tensors="pt").to(model.device)

        if args.layer_count > 0:
            ori_out = model(**test_input)

        if args.rot:
            rot_model(model)

        if args.layer_count > 0:
            rot_out = model(**test_input)
            loss = torch.nn.MSELoss()
            logger.info(loss(ori_out[0], rot_out[0]))

    with torch.no_grad():
        anti_config = AntiOutlierConfig(w_bit=8,
                                        a_bit=8,
                                        anti_method='m4',
                                        dev_type='npu',
                                        dev_id=model.device.index)
        anti_outlier = AntiOutlier(model, calib_data=anti_dataset, cfg=anti_config)
        anti_outlier.process()

    disable_names = []
    for ids in range(config.num_hidden_layers):
        disable_names.append("model.layers." + str(ids) + ".mlp.gate")

    quant_config = QuantConfig(
        a_bit=8,
        w_bit=8,
        disable_names=disable_names,
        dev_type='npu',
        dev_id=model.device.index,
        act_method=1,
        pr=1.0,
        w_sym=True,
        mm_tensor=False,
    )

    calibrator = Calibrator(model,
                            quant_config,
                            calib_data=dataset_calib,
                            disable_level="L0",
                            mix_cfg={
                                "*.mlp.*": "w8a8_dynamic",
                                "*": "w8a8"
                            })
    calibrator.run()

    if args.mindie_format:
        quant_model_description_json_name = "quant_model_description_w8a8_dynamic.json"
    else:
        quant_model_description_json_name = "quant_model_description.json"

    save_type = "safe_tensor" if args.mindie_format else "ascendV1"
    calibrator.save(save_path,
                    json_name=quant_model_description_json_name,
                    safetensors_name="quant_model_weight_w8a8_dynamic.safetensors",
                    save_type=[save_type],
                    part_file_size=4)

    custom_hooks = {
        'config.json': functools.partial(modify_config_json, custom_hook=custom_hook)
    }
    copy_config_files(input_path=model_path, output_path=save_path, quant_config=quant_config,
                      mindie_format=args.mindie_format, custom_hooks=custom_hooks)


if __name__ == "__main__":
    # torch_npu will fork a new process to init,
    # it's lazy_init will fail after we load a big model,so we need to init it here
    torch_npu.npu.init()
    # Invoke main process
    main()
