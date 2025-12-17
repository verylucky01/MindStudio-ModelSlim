#  -*- coding: utf-8 -*-
#  Copyright (c) 2025-2025 Huawei Technologies Co., Ltd.
#  #
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#  #
#  http://www.apache.org/licenses/LICENSE-2.0
#  #
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

import os
import sys
import time
from pprint import pformat

import colossalai
from tqdm import tqdm
import torch
import torch.distributed as dist
from safetensors import safe_open
from thop import clever_format, profile
from colossalai.utils import get_current_device

sys.path.append(f"{os.environ['PROJECT_PATH']}/resource/multi_modal/opensora_project")

from opensora.datasets import save_sample
from opensora.datasets.aspect import get_image_size, get_num_frames
from opensora.acceleration.parallel_mgr import set_parallel_manager
from opensora.models.text_encoder.t5 import text_preprocessing
from opensora.registry import MODELS, SCHEDULERS, build_module
from opensora.utils.config_utils import parse_configs
from opensora.utils.misc import all_exists, create_logger, is_distributed, is_main_process, to_torch_dtype
from opensora.utils.inference_utils import (
    add_watermark,
    append_generated,
    append_score_to_prompts,
    apply_mask_strategy,
    collect_references_batch,
    dframe_to_frame,
    extract_json_from_prompts,
    extract_prompts_loop,
    get_save_path_name,
    load_prompts,
    merge_prompt,
    prepare_multi_resolution_info,
    refine_prompts_by_openai,
    split_prompt,
)

from mmengine.runner import set_random_seed

from msmodelslim.pytorch.quant.ptq_tools import QuantConfig, Calibrator

@torch.no_grad()
def main():
    # 1.cfg and init distrubuted env
    cfg = parse_configs(False)
    print(cfg)

    # 2.Initialize Distributed Training
    colossalai.launch_from_torch({}, seed=cfg.seed)
    device = get_current_device()
    dtype = to_torch_dtype(cfg.dtype)

    # 3.Initialize Process Group
    sp_size = 1
    dp_size = 1
    enable_sequence_parallelism = False
    set_parallel_manager(sp_size, dp_size, dp_axis=0, sp_axis=1)

    # 4.runtime variables
    torch.set_grad_enabled(False)
    device = "npu" if torch.npu.is_available() else "cpu"
    device_0, device_1 = torch.device("npu:0"), torch.device("npu:0")
    dtype = to_torch_dtype(cfg.dtype)
    prompts = cfg.prompt

    # init logger
    logger = create_logger()
    logger.info("Inference configuration:\n %s", pformat(cfg.to_dict()))
    verbose = cfg.get("verbose", 2)
    progress_wrap = tqdm

    # 5.build mode & load weights
    logger.info("Building model...")

    model_cfg = cfg.model
    vae_cfg = cfg.vae
    text_encoder_cfg = cfg.text_encoder

    model_cfg['from_pretrained'] = f"{os.environ['PROJECT_PATH']}/resource/multi_modal/" + \
        "opensora_project/open-sora/transformer/"
    vae_cfg['from_pretrained'] = f"{os.environ['PROJECT_PATH']}/resource/multi_modal/opensora_project/open-sora/vae/"
    text_encoder_cfg['from_pretrained'] = f"{os.environ['PROJECT_PATH']}/resource/multi_modal/" + \
    "opensora_project/open-sora/text_encoder/"

    text_encoder = build_module(text_encoder_cfg, MODELS, device=device_1)
    text_encoder.t5.model.to(dtype).eval()
    vae = build_module(vae_cfg, MODELS).to(device_1, dtype).eval()
    image_size = cfg.get("images_size", None)
    if image_size is None:
        resolution = cfg.get("resolution", None)
        aspect_ratio = cfg.get("aspect_ratio", None)
        assert (
            resolution is not None and aspect_ratio is not None
        ), "resolution and aspect_ratio must be provided if images_size is not provided"
        image_size = get_image_size(resolution, aspect_ratio)
    num_frames = get_num_frames(cfg.num_frames)

    input_size = (num_frames, *image_size)
    latent_size = vae.get_latent_size(input_size)
    model = (
        build_module(
            model_cfg,
            MODELS,
            input_size=latent_size,
            in_channels=vae.out_channels,
            caption_channels=text_encoder.output_dim,
            model_max_length=text_encoder.model_max_length,
            enable_sequence_parallelism=enable_sequence_parallelism
        ).to(device_0, dtype).eval()
    )
    text_encoder.y_embedder = model.y_embedder

    # buiild scheduler
    scheduler = build_module(cfg.scheduler, SCHEDULERS)

    ori_fp_weight = {}
    for key, value in model.state_dict().items():
        if not isinstance(value, torch.Tensor):
            continue
        ori_fp_weight[key] = value
    
    calib_dataset = torch.load(f"{os.environ['PROJECT_PATH']}/resource/multi_modal/opensora_project/" + \
                               "opensora_calib_v2.pth", map_location="npu")
    
    # quantization config
    q_config = QuantConfig(
        w_bit=8,
        a_bit=8,
        w_signed=True,
        a_signed=True,
        w_sym=True,
        a_sym=False,
        act_quant=True,
        act_method=1,
        quant_mode=1,
        disable_names=None,
        amp_num=0,
        keep_acc=None,
        sigma=25,
        device="npu"
    )

    calibrator = Calibrator(model, q_config, calib_data=calib_dataset[:1])
    calibrator.run()
    calibrator.export_quant_safetensor(f"{os.environ['PROJECT_PATH']}/output/ptq-tools/quant_opensora_1_2")

if __name__ == "__main__":
    main()
