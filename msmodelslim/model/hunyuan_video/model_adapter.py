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
import logging
import os
import re
import random
import sys
import time
from importlib import import_module
from pathlib import Path
from typing import Optional, Dict, Any, Tuple, Generator, List, Callable

import torch
from torch import nn, distributed as dist
from tqdm import tqdm

from msmodelslim.core.const import DeviceType
from msmodelslim.core.base.protocol import ProcessRequest
from msmodelslim.model.base import BaseModelAdapter
from msmodelslim.model.common.layer_wise_forward import TransformersForwardBreak, \
    generated_decoder_layer_visit_func_with_keyword
from msmodelslim.utils.cache import to_device
from msmodelslim.utils.exception import InvalidModelError, SchemaValidateError, UnsupportedError
from msmodelslim.utils.logging import logger_setter, get_logger
from ..interface_hub import ModelInfoInterface, MultimodalSDPipelineInterface, FA3QuantAdapterInterface, FA3QuantPlaceHolder, \
    OnlineQuaRotInterface

SUPPORTED_TASKS = ['hunyuan_video']


@logger_setter()
class HunyuanVideoModelAdapter(BaseModelAdapter,
                          ModelInfoInterface,
                          MultimodalSDPipelineInterface,
                          FA3QuantAdapterInterface,
                          OnlineQuaRotInterface,
                          ):
    def __init__(self,
                 model_type: str,
                 model_path: Path,
                 trust_remote_code: bool = False):
        super().__init__(model_type, model_path, trust_remote_code)
        self.pipeline = None
        self.transformer = None
        self.model_args = None

        self._get_default_model_args()

    def get_model_type(self) -> str:
        return self.model_type
    
    def get_model_pedigree(self) -> str:
        return 'hunyuan_video'
    
    def handle_dataset(self, dataset: Any, device: DeviceType = DeviceType.NPU) -> Generator[Any, None, None]:
        return dataset
    
    def init_model(self, device: DeviceType = DeviceType.NPU) -> Dict[str, nn.Module]:
        return {'': self.transformer}
    
    def generate_model_forward(self, model: torch.nn.Module,
                               inputs: Any,
                               ) -> Generator[ProcessRequest, Any, None]:
        transformer_blocks = [
            (name, module)
            for name, module in model.named_modules()
            if "streamblock" in module.__class__.__name__.lower()
        ]
        first_block_input = None

        def break_hook(module: nn.Module, hook_args: Tuple[Any, ...], hook_kwargs: Dict[str, Any]):
            nonlocal first_block_input
            first_block_input = (hook_args, hook_kwargs,)
            raise TransformersForwardBreak()

        hooks = [transformer_blocks[0][1].register_forward_pre_hook(break_hook, with_kwargs=True)]

        try:
            if isinstance(inputs, list) or isinstance(inputs, tuple):
                model(*inputs)
            elif isinstance(inputs, dict):
                model(**inputs)
            else:
                model(inputs)
        except TransformersForwardBreak:
            pass
        except Exception as e:
            raise e
        finally:
            for hook in hooks:
                hook.remove()

        if first_block_input is None:
            raise InvalidModelError("Can't get first block input.", action="Please check the model and input")

        first_block_input = to_device(first_block_input, 'cpu')
        current_inputs = first_block_input

        if dist.is_initialized():
            dist.barrier()

        for name, block in transformer_blocks:
            args, kwargs = current_inputs
            outputs = yield ProcessRequest(name, block, args, kwargs)
            hidden_states = outputs
            current_inputs = ((hidden_states,), current_inputs[1])

    def generate_model_visit(self, model: torch.nn.Module,
                             transformer_blocks: Optional[List[Tuple[str, torch.nn.Module]]] = None,
                             ) -> Generator[ProcessRequest, Any, None]:
        return generated_decoder_layer_visit_func_with_keyword(model, keyword="streamblock")

    def enable_kv_cache(self, model: nn.Module, need_kv_cache: bool) -> None:
        pass

    def run_calib_inference(self):
        """运行校准推理"""
        stream = torch.npu.Stream()
        args = self.model_args
        prompt = self.model_args.prompt
        # Start sampling
        for _ in tqdm(range(1), desc='Dump calib data by float model inference'):
            begin = time.time()
            outputs = self.hunyuan_video_sampler.predict(
            prompt=prompt, 
            height=args.video_size[0],
            width=args.video_size[1],
            video_length=args.video_length,
            seed=args.seed,
            negative_prompt=args.neg_prompt,
            infer_steps=args.infer_steps,
            guidance_scale=args.cfg_scale,
            num_videos_per_prompt=args.num_videos,
            flow_shift=args.flow_shift,
            batch_size=args.batch_size,
            embedded_guidance_scale=args.embedded_cfg_scale
        )
        stream.synchronize()
        end = time.time()
        logging.info(f"Generating video used time {end - begin: .4f}s")

    def apply_quantization(self, process_model_func):
        from contextlib import contextmanager
        import torch.cuda.amp as amp

        @contextmanager
        def noop_no_sync():
            yield

        no_sync = getattr(self, 'no_sync', noop_no_sync)

        for name, module in self.transformer.named_modules():
            if 'blocks' not in name:
                module.to('npu')
            else:
                module.to('cpu')
        with amp.autocast(dtype=torch.bfloat16), torch.no_grad(), no_sync():
            process_model_func()

    def load_pipeline(self):
        self._load_pipeline()
        self._setup_cache()

    def set_model_args(self, override_model_config: object):
        self.model_args.model_base = self.model_path
        self.model_args.dit_weight = os.path.join(self.model_args.model_base, 
            "hunyuan-video-t2v-720p",
            "transformers",
            "mp_rank_00_model_states.pt"
        )
        self.model_args.vae_path = os.path.join(self.model_args.model_base, 
            "hunyuan-video-t2v-720p",
            "vae"
        )
        self.model_args.text_encoder_path = os.path.join(self.model_args.model_base, 
            "text_encoder"
        )
        self.model_args.text_encoder_2_path = os.path.join(self.model_args.model_base, 
            "clip-vit-large-patch14"
        )

        missing_attrs = []
        for key in override_model_config.keys():
            if not hasattr(self.model_args, key):
                missing_attrs.append(key)

        if missing_attrs:
            available = [a for a in dir(self.model_args)]
            raise SchemaValidateError(
                f"illegal config attributes: {missing_attrs}. \n"
                f"supported config attributes: {available}"
            )
        
        for key in override_model_config.keys():
            setattr(self.model_args, key, override_model_config[key])

        parser = self._get_parser()
        argv = []
        for key, val in vars(self.model_args).items():
            if val is None:
                continue
            elif key == "video_size":
                continue
            elif isinstance(val, bool):
                if val:
                    argv.append(f"--{key}")
            else:
                argv.extend([f"--{key}", str(val)])

        self.model_args = parser.parse_args(argv)
        self.model_args.latent_channels = int(self.model_args.latent_channels)
        self.model_args = self.__sanity_check_args(self.model_args)

        self._validate_args(self.model_args)

    def _get_default_model_args(self):
        parser = self._get_parser()
        args = parser.parse_args([])
        self.model_args = args

    def _get_parser(self) -> argparse.ArgumentParser:
        self._check_import_dependency()
        parser = argparse.ArgumentParser(description="HunyuanVideo inference script")

        parser = self.__add_device_args(parser)
        parser = self.__add_network_args(parser)
        parser = self.__add_extra_models_args(parser)
        parser = self.__add_denoise_schedule_args(parser)
        parser = self.__add_inference_args(parser)
        parser = self.__add_parallel_args(parser)
        parser = self.__add_ditcache_args(parser)
        parser = self.__add_attentioncache_args(parser)
        parser = self.__add_quant_args(parser)

        return parser

    def _check_import_dependency(self):
        try:
            import hyvideo
            from hyvideo.constants import PRECISION_TO_TYPE, C_SCALE, PROMPT_TEMPLATE_ENCODE, \
                PROMPT_TEMPLATE_ENCODE_VIDEO, NEGATIVE_PROMPT, PROMPT_TEMPLATE, PRECISIONS, \
                    NORMALIZATION_TYPE, ACTIVATION_TYPE, MODEL_BASE, DATA_TYPE, VAE_PATH, \
                        TEXT_ENCODER_PATH, TOKENIZER_PATH, TEXT_PROJECTION
            from hyvideo.modules.models import HUNYUAN_VIDEO_CONFIG
            from hyvideo.inference import HunyuanVideoSampler
            from hyvideo.utils.file_utils import save_videos_grid

        except ImportError as e:
        # Concise import error message
            raise ImportError(
                "Failed to import required components from hunyuanvideo. "
                "Please install the hunyuanvideo dependencies from the official source, "
                "make sure you can run the original floating-point inference successfully, "
                "and add the hunyuanvideo repository to the Python search path environment variable PYTHONPATH. "
                "e.g. export PYTHONPATH=/path/to/hunyuanvideo:$PYTHONPATH"
            ) from e

    def _validate_args(self, args):
        """Get default parameter configuration, integrating wan config parameters"""
        self._check_import_dependency()
        models_root_path = Path(args.model_base)
        if not models_root_path.exists():
            raise SchemaValidateError(f"`model_base` not exists : {args.model_base}")
        args.task_config = 'hunyuanvideo'
        # create save folder to save the samples
        save_path = args.save_path if not args.save_path_suffix else f'{args.save_path}_{args.save_path_suffix}'
        os.makedirs(save_path, exist_ok=True)

        if args.infer_steps is None:
            args.infer_steps = 50
        if not isinstance(args.infer_steps, int):
            raise SchemaValidateError(
                f"sample_steps must be an integer, got {type(args.infer_steps).__name__}"
            )
        if args.infer_steps <= 0:
            raise SchemaValidateError(f"sample_steps mush be greater than 0")

        if args.batch_size is None:
            args.batch_size = 1
        if not isinstance(args.batch_size, int):
            raise SchemaValidateError(
                f"batch_size must be an integer, got {type(args.batch_size).__name__}"
            )
        if args.batch_size <= 0:
            raise SchemaValidateError(f"batch_size must be greater than 0")
        if args.seed is None:
            args.seed = 0
        args.seed = args.seed if args.seed >= 0 else random.randint(0, sys.maxsize)

        # Validate prompt
        prompt = getattr(args, "prompt", None)
        if prompt is None:
            raise SchemaValidateError("Missing required parameter: prompt")
        if not isinstance(args.prompt, str):
            raise SchemaValidateError(f"prompt must be a string, got {type(args.prompt).__name__}")
        if not args.prompt.strip():
            raise SchemaValidateError("prompt cannot be an empty string")

    def _setup_cache(self):
        # 设置Cache机制
        try:
            from mindiesd import CacheConfig, CacheAgent
        except ImportError as e:
            # Concise import error message
            raise ImportError(
                "Failed to import required components from mindiesd. "
            ) from e
        args = self.model_args
        if args.use_cache and len(self.transformer.single_blocks) > 0:
            # single
            config_single = CacheConfig(
                method="dit_block_cache",
                blocks_count=len(self.transformer.single_blocks),
                steps_count=args.infer_steps,
                step_start=args.cache_start_steps,
                step_interval=args.cache_interval,
                step_end=args.infer_steps - 1,
                block_start=args.single_block_start,
                block_end=args.single_block_end
            )
            cache_single = CacheAgent(config_single)
            self.transformer.cache_single = cache_single
        if args.use_cache_double and len(self.transformer.double_blocks) > 0:
            # double
            config_double = CacheConfig(
                method="dit_block_cache",
                blocks_count=len(self.transformer.double_blocks),
                steps_count=args.infer_steps,
                step_start=args.cache_start_steps,
                step_interval=args.cache_interval,
                step_end=args.infer_steps - 1,
                block_start=args.double_block_start,
                block_end=args.double_block_end
            )
            cache_dual = CacheAgent(config_double)
            self.transformer.cache_dual = cache_dual

        if args.use_attentioncache:
            if len(self.transformer.double_blocks) > 0:
                config_double = CacheConfig(
                    method="attention_cache",
                    blocks_count=len(self.transformer.double_blocks),
                    steps_count=args.infer_steps,
                    step_start=args.start_step,
                    step_interval=args.attentioncache_interval,
                    step_end=args.end_step
                )
            if len(self.transformer.single_blocks) > 0:
                config_single = CacheConfig(
                    method="attention_cache",
                    blocks_count=len(self.transformer.single_blocks),
                    steps_count=args.infer_steps,
                    step_start=args.start_step,
                    step_interval=args.attentioncache_interval,
                    step_end=args.end_step
                )
        else:
            if len(self.transformer.double_blocks) > 0:
                config_double = CacheConfig(
                    method="attention_cache",
                    blocks_count=len(self.transformer.double_blocks),
                    steps_count=args.infer_steps
                )
            if len(self.transformer.single_blocks) > 0:
                config_single = CacheConfig(
                    method="attention_cache",
                    blocks_count=len(self.transformer.single_blocks),
                    steps_count=args.infer_steps
                )
        if len(self.transformer.double_blocks) > 0:
            cache_double = CacheAgent(config_double)
            for block in self.transformer.double_blocks:
                block.cache = cache_double
        if len(self.transformer.single_blocks) > 0:
            cache_single = CacheAgent(config_single)
            for block in self.transformer.single_blocks:
                block.cache = cache_single

    def _load_pipeline(self):
        self._check_import_dependency()

        import hyvideo
        from hyvideo.inference import HunyuanVideoSampler

        args = self.model_args
        if args.ulysses_degree > 1 or args.ring_degree > 1:
            raise UnsupportedError("context parallel are not supported in non-distributed environments")
        if args.vae_parallel:
            raise UnsupportedError("vae parallel are not support in non-distributed environment")

        logging.info("load hunyuan_video models")
        models_root_path = Path(args.model_base)
        self.hunyuan_video_sampler = HunyuanVideoSampler.from_pretrained(models_root_path, args=args)
        self.transformer = self.hunyuan_video_sampler.pipeline.transformer

    def __add_device_args(self, parser: argparse.ArgumentParser):
        group = parser.add_argument_group(title="HunyuanVideo device args")

        group.add_argument(
            "--device_id",
            type=int,
            default=0
        )
        return parser

    def __add_network_args(self, parser: argparse.ArgumentParser):
        group = parser.add_argument_group(title="HunyuanVideo network args")
        from hyvideo.constants import PRECISIONS
        from hyvideo.modules.models import HUNYUAN_VIDEO_CONFIG
        # Main model
        group.add_argument(
            "--model",
            type=str,
            choices=list(HUNYUAN_VIDEO_CONFIG.keys()),
            default="HYVideo-T/2-cfgdistill",
        )
        group.add_argument(
            "--latent_channels",
            type=str,
            default=16,
            help="Number of latent channels of DiT. If None, it will be determined by `vae`. If provided, "
            "it still needs to match the latent channels of the VAE model.",
        )
        group.add_argument(
            "--precision",
            type=str,
            default="bf16",
            choices=PRECISIONS,
            help="Precision mode. Options: fp32, fp16, bf16. Applied to the backbone model and optimizer.",
        )

        # RoPE
        group.add_argument(
            "--rope_theta", type=int, default=256, help="Theta used in RoPE."
        )
        return parser

    def __add_extra_models_args(self, parser: argparse.ArgumentParser):
        group = parser.add_argument_group(
            title="Extra models args, including vae, text encoders and tokenizers)"
        )
        from hyvideo.constants import PROMPT_TEMPLATE, PRECISIONS, \
            VAE_PATH, TEXT_ENCODER_PATH, TOKENIZER_PATH
        # - VAE
        group.add_argument(
            "--vae_parallel",
            action="store_true",
            help="Use vae parallel",
        )

        group.add_argument(
            "--vae_path",
            type=str,
            default="vae",
            help="Path of VAE model",
        )
        group.add_argument(
            "--vae",
            type=str,
            default="884-16c-hy",
            choices=list(VAE_PATH),
            help="Name of the VAE model.",
        )
        group.add_argument(
            "--vae_precision",
            type=str,
            default="fp16",
            choices=PRECISIONS,
            help="Precision mode for the VAE model.",
        )
        group.add_argument(
            "--vae_tiling",
            action="store_true",
            help="Enable tiling for the VAE model to save GPU memory.",
        )
        group.set_defaults(vae_tiling=True)
        logging.info(f"Enable tiling for the VAE model to save GPU memory.")
        group.add_argument(
            "--text_encoder_path",
            type=str,
            default="text_encoder",
            help="Path of text encoder model",
        )
        group.add_argument(
            "--text_encoder",
            type=str,
            default="llm",
            choices=list(TEXT_ENCODER_PATH),
            help="Name of the text encoder model.",
        )
        group.add_argument(
            "--text_encoder_precision",
            type=str,
            default="fp16",
            choices=PRECISIONS,
            help="Precision mode for the text encoder model.",
        )
        group.add_argument(
            "--text_states_dim",
            type=int,
            default=4096,
            help="Dimension of the text encoder hidden states.",
        )
        group.add_argument(
            "--text_len", type=int, default=256, help="Maximum length of the text input."
        )
        group.add_argument(
            "--tokenizer",
            type=str,
            default="llm",
            choices=list(TOKENIZER_PATH),
            help="Name of the tokenizer model.",
        )
        group.add_argument(
            "--prompt_template",
            type=str,
            default="dit-llm-encode",
            choices=PROMPT_TEMPLATE,
            help="Image prompt template for the decoder-only text encoder model.",
        )
        group.add_argument(
            "--prompt_template_video",
            type=str,
            default="dit-llm-encode-video",
            choices=PROMPT_TEMPLATE,
            help="Video prompt template for the decoder-only text encoder model.",
        )
        group.add_argument(
            "--hidden_state_skip_layer",
            type=int,
            default=2,
            help="Skip layer for hidden states.",
        )
        group.add_argument(
            "--apply_final_norm",
            action="store_true",
            help="Apply final normalization to the used text encoder hidden states.",
        )

        # - CLIP
        group.add_argument(
            "--text_encoder_2_path",
            type=str,
            default="clip-vit-large-patch14",
            help="Path of text encoder model",
        )
        group.add_argument(
            "--text_encoder_2",
            type=str,
            default="clipL",
            choices=list(TEXT_ENCODER_PATH),
            help="Name of the second text encoder model.",
        )
        group.add_argument(
            "--text_encoder_precision_2",
            type=str,
            default="fp16",
            choices=PRECISIONS,
            help="Precision mode for the second text encoder model.",
        )
        group.add_argument(
            "--text_states_dim_2",
            type=int,
            default=768,
            help="Dimension of the second text encoder hidden states.",
        )
        group.add_argument(
            "--tokenizer_2",
            type=str,
            default="clipL",
            choices=list(TOKENIZER_PATH),
            help="Name of the second tokenizer model.",
        )
        group.add_argument(
            "--text_len_2",
            type=int,
            default=77,
            help="Maximum length of the second text input.",
        )

        return parser

    def __add_denoise_schedule_args(self, parser: argparse.ArgumentParser):
        group = parser.add_argument_group(title="Denoise schedule args")

        group.add_argument(
            "--denoise_type",
            type=str,
            default="flow",
            help="Denoise type for noised inputs.",
        )

        # Flow Matching
        group.add_argument(
            "--flow_shift",
            type=float,
            default=7.0,
            help="Shift factor for flow matching schedulers.",
        )
        group.add_argument(
            "--flow_reverse",
            action="store_true",
            help="If reverse, learning/sampling from t=1 -> t=0.",
        )
        group.add_argument(
            "--flow_solver",
            type=str,
            default="euler",
            help="Solver for flow matching.",
        )
        group.add_argument(
            "--use_linear_quadratic_schedule",
            action="store_true",
            help="Use linear quadratic schedule for flow matching."
            "Following MovieGen (https://ai.meta.com/static-resource/movie-gen-research-paper)",
        )
        group.add_argument(
            "--linear_schedule_end",
            type=int,
            default=25,
            help="End step for linear quadratic schedule for flow matching.",
        )

        return parser

    def __add_inference_args(self, parser: argparse.ArgumentParser):
        group = parser.add_argument_group(title="Inference args")

        # ======================== Model loads ========================
        group.add_argument(
            "--model_base",
            type=str,
            default="ckpts",
            help="Root path of all the models, including t2v models and extra models.",
        )
        group.add_argument(
            "--dit_weight",
            type=str,
            default="ckpts/hunyuan-video-t2v-720p/transformers/mp_rank_00_model_states.pt",
            help="Path to the HunyuanVideo model. If None, search the model in the args.model_root."
            "1. If it is a file, load the model directly."
            "2. If it is a directory, search the model in the directory. Support two types of models: "
            "1) named `pytorch_model_*.pt`"
            "2) named `*_model_states.pt`, where * can be `mp_rank_00`.",
        )
        group.add_argument(
            "--model_resolution",
            type=str,
            default="540p",
            choices=["540p", "720p"],
            help="Root path of all the models, including t2v models and extra models.",
        )
        group.add_argument(
            "--load_key",
            type=str,
            default="module",
            help="Key to load the model states. 'module' for the main model, 'ema' for the EMA model.",
        )
        group.add_argument(
            "--use_cpu_offload",
            action="store_true",
            help="Use CPU offload for the model load.",
        )

        # ======================== Inference general setting ========================
        group.add_argument(
            "--batch_size",
            type=int,
            default=1,
            help="Batch size for inference and evaluation.",
        )
        group.add_argument(
            "--infer_steps",
            type=int,
            default=50,
            help="Number of denoising steps for inference.",
        )
        group.add_argument(
            "--disable_autocast",
            action="store_true",
            help="Disable autocast for denoising loop and vae decoding in pipeline sampling.",
        )
        group.add_argument(
            "--save_path",
            type=str,
            default="./results",
            help="Path to save the generated samples.",
        )
        group.add_argument(
            "--save_path_suffix",
            type=str,
            default="",
            help="Suffix for the directory of saved samples.",
        )
        group.add_argument(
            "--name_suffix",
            type=str,
            default="",
            help="Suffix for the names of saved samples.",
        )
        group.add_argument(
            "--num_videos",
            type=int,
            default=1,
            help="Number of videos to generate for each prompt.",
        )
        # ---sample size---
        group.add_argument(
            "--video_size",
            type=int,
            nargs="+",
            default=(720, 1280),
            help="Video size for training. If a single value is provided, it will be used for both height "
            "and width. If two values are provided, they will be used for height and width "
            "respectively.",
        )
        group.add_argument(
            "--video_length",
            type=int,
            default=129,
            help="How many frames to sample from a video. if using 3d vae, the number should be 4n+1",
        )
        # --- prompt ---
        group.add_argument(
            "--prompt",
            type=str,
            default=None,
            help="Prompt for sampling during evaluation.",
        )
        group.add_argument(
            "--seed_type",
            type=str,
            default="auto",
            choices=["file", "random", "fixed", "auto"],
            help="Seed type for evaluation. If file, use the seed from the CSV file. If random, generate a "
            "random seed. If fixed, use the fixed seed given by `--seed`. If auto, `csv` will use the "
            "seed column if available, otherwise use the fixed `seed` value. `prompt` will use the "
            "fixed `seed` value.",
        )
        group.add_argument("--seed", type=int, default=None, help="Seed for evaluation.")

        # Classifier-Free Guidance
        group.add_argument(
            "--neg_prompt", type=str, default=None, help="Negative prompt for sampling."
        )
        group.add_argument(
            "--cfg_scale", type=float, default=1.0, help="Classifier free guidance scale."
        )
        group.add_argument(
            "--embedded_cfg_scale",
            type=float,
            default=6.0,
            help="Embeded classifier free guidance scale.",
        )

        group.add_argument(
            "--use_fp8",
            action="store_true",
            help="Enable use fp8 for inference acceleration."
        )

        group.add_argument(
            "--reproduce",
            action="store_true",
            help="Enable reproducibility by setting random seeds and deterministic algorithms.",
        )

        return parser

    def __add_parallel_args(self, parser: argparse.ArgumentParser):
        group = parser.add_argument_group(title="Parallel args")

        # ======================== Model loads ========================
        group.add_argument(
            "--ulysses_degree",
            type=int,
            default=1,
            help="Ulysses degree.",
        )
        group.add_argument(
            "--ring_degree",
            type=int,
            default=1,
            help="Ulysses degree.",
        )

        return parser

    def __add_ditcache_args(self, parser: argparse.ArgumentParser):
        group = parser.add_argument_group(title="Dit Cache args")
        
        # single cache related config
        group.add_argument("--use_cache", action='store_true')
        group.add_argument("--cache_interval", type=int, default=3)
        group.add_argument("--cache_start_steps", type=int, default=10)

        group.add_argument("--single_block_start", type=int, default=5)
        group.add_argument("--single_block_end", type=int, default=35)

        ## double stream cache related config
        group.add_argument("--use_cache_double", action='store_true')
        group.add_argument("--double_block_start", type=int, default=3)
        group.add_argument("--double_block_end", type=int, default=18)

        # cache searcher config
        group.add_argument("--search_single_cache", action='store_true')
        group.add_argument("--search_double_cache", action='store_true')
        group.add_argument("--cache_ratio", type=float, default=1.2)

        return parser

    def __add_attentioncache_args(self, parser: argparse.ArgumentParser):
        group = parser.add_argument_group(title="Attention Cache args")

        group.add_argument("--use_attentioncache", action='store_true')
        group.add_argument("--attentioncache_ratio", type=float, default=1.2)
        group.add_argument("--attentioncache_interval", type=int, default=3)
        group.add_argument("--start_step", type=int, default=9)
        group.add_argument("--end_step", type=int, default=47)

        return parser

    def __add_quant_args(self, parser: argparse.ArgumentParser):
        group = parser.add_argument_group(title="Quant args")
        group.add_argument(
            "--quant_desc_path",
            type=str,
            help="Path to quantization description file (enables quantization if specified, \
                format: quant_model_description_*.json)"
        )
        return parser

    def __sanity_check_args(self, args):
        # VAE channels
        vae_pattern = r"\d{2,3}-\d{1,2}c-\w+"
        if not re.match(vae_pattern, args.vae):
            raise SchemaValidateError(
                f"Invalid VAE model: {args.vae}. Must be in the format of '{vae_pattern}'."
            )
        vae_channels = int(args.vae.split("-")[1][:-1])
        if args.latent_channels is None:
            args.latent_channels = vae_channels
        if vae_channels != args.latent_channels:
            raise SchemaValidateError(
                f"Latent channels ({args.latent_channels}) must match the VAE channels ({vae_channels})."
            )
        return args

    # ===== OnlineQuaRotInterface =====
    def get_online_rotation_configs(self, model: Optional[nn.Module] = None):
        """
        返回在线旋转配置，配置 q_rot 和 k_rot 为旋转矩阵替换。
        
        如果提供了 model，会在此方法中直接给 MMDoubleStreamBlock 和 MMSingleStreamBlock 挂载 q_rot 和 k_rot Identity 模块。
        
        Args:
            model: 可选的模型实例，如果提供，会在此方法中挂载 Identity 模块
        
        Returns:
            Dict[str, RotationConfig]: 模块名到旋转配置的映射
        """
        configs = {}
        
        # 如果提供了 model，直接挂载 Identity 模块
        if model is not None:
            for name, module in model.named_modules():
                module_type = module.__class__.__name__
                
                # 只处理目标模块类型
                if module_type not in ["MMDoubleStreamBlock", "MMSingleStreamBlock"]:
                    continue
                
                try:
                    # 创建并挂载 q_rot 和 k_rot Identity 模块
                    if not hasattr(module, 'q_rot'):
                        module.register_module('q_rot', nn.Identity())
                    if not hasattr(module, 'k_rot'):
                        module.register_module('k_rot', nn.Identity())
                    get_logger().debug(f"Registered q_rot and k_rot Identity modules for {name}")
                except Exception as e:
                    get_logger().warning(f"Failed to register rotation modules for {name}: {str(e)}")
        
        # 配置旋转，q_rot 和 k_rot 使用相同的随机数种子，确保生成相同的旋转矩阵
        shared_seed = 1234  # q_rot 和 k_rot 共享的随机数种子
        
        # 遍历模型找到所有目标模块并配置旋转
        target_model = model if model is not None else getattr(self, 'transformer', None)
        if target_model is None:
            get_logger().warning("No model provided and transformer not available, returning empty rotation configs")
            return configs
        
        # 获取全局 head_dim - 从 transformer 直接获取
        if not hasattr(target_model, 'hidden_size') or not hasattr(target_model, 'heads_num'):
            get_logger().warning("Could not determine head_dim from transformer, returning empty rotation configs")
            return configs
        
        head_dim = target_model.hidden_size // target_model.heads_num

        # 使用全局 head_dim 为所有目标模块配置旋转
        for name, module in target_model.named_modules():
            module_type = module.__class__.__name__
            
            # 只处理目标模块类型
            if module_type not in ["MMDoubleStreamBlock", "MMSingleStreamBlock"]:
                continue
            
            # 配置 q_rot
            q_rot_path = f"{name}.q_rot" if name else "q_rot"
            configs[q_rot_path] = OnlineQuaRotInterface.RotationConfig(
                rotation_type="replace",
                rotation_size=head_dim,
                rotation_mode=OnlineQuaRotInterface.QuaRotMode.HADAMARD,
                block_size=-1,
                seed=shared_seed,
                dtype=torch.bfloat16
            )
            
            # 配置 k_rot（使用相同的种子，确保与 q_rot 使用相同的旋转矩阵）
            k_rot_path = f"{name}.k_rot" if name else "k_rot"
            configs[k_rot_path] = OnlineQuaRotInterface.RotationConfig(
                rotation_type="replace",
                rotation_size=head_dim,
                rotation_mode=OnlineQuaRotInterface.QuaRotMode.HADAMARD,
                block_size=-1,
                seed=shared_seed,
                dtype=torch.bfloat16
            )
        
        return configs

    # ===== FA3QuantAdapterInterface =====
    def inject_fa3_placeholders(self, root_name: str, root_module: nn.Module, should_inject: Callable[[str], bool]) -> None:
        """为 HunyuanVideo 模型的 MMDoubleStreamBlock 和 MMSingleStreamBlock 安装 FA3 占位，并包裹 forward 调用这些占位。

        - 在每个目标模块下注入子模块：fa3_q, fa3_k, fa3_v
        - 包裹其 forward 方法，在计算 Q、K、V 并 cat 后，依次调用占位：
            q = self.fa3_q(q)
            k = self.fa3_k(k)
            v = self.fa3_v(v)
        """

        def _wrap_double_forward(module: nn.Module):
            """包裹 MMDoubleStreamBlock 的 forward 方法"""
            original_forward = module.forward
            
            # 动态导入必要的函数
            hyvideo_double_module = import_module(original_forward.__module__)
            modulate = hyvideo_double_module.modulate
            apply_gate = hyvideo_double_module.apply_gate
            rearrange = hyvideo_double_module.rearrange
            apply_rotary_emb = hyvideo_double_module.apply_rotary_emb
            attention = hyvideo_double_module.attention
            parallel_attention = hyvideo_double_module.parallel_attention

            def new_forward(
                    self,
                    img: torch.Tensor,
                    txt: torch.Tensor,
                    vec: torch.Tensor,
                    cu_seqlens_q: Optional[torch.Tensor] = None,
                    cu_seqlens_kv: Optional[torch.Tensor] = None,
                    max_seqlen_q: Optional[int] = None,
                    max_seqlen_kv: Optional[int] = None,
                    freqs_cis: tuple = None,
                ) -> Tuple[torch.Tensor, torch.Tensor]:
                # 从 vec 中提取 modulation 参数
                (
                    img_mod1_shift,
                    img_mod1_scale,
                    img_mod1_gate,
                    img_mod2_shift,
                    img_mod2_scale,
                    img_mod2_gate,
                ) = self.img_mod(vec).chunk(6, dim=-1)
                (
                    txt_mod1_shift,
                    txt_mod1_scale,
                    txt_mod1_gate,
                    txt_mod2_shift,
                    txt_mod2_scale,
                    txt_mod2_gate,
                ) = self.txt_mod(vec).chunk(6, dim=-1)
                # Prepare image for attention.
                img_modulated = self.img_norm1(img)
                img_modulated = modulate(
                    img_modulated, shift=img_mod1_shift, scale=img_mod1_scale
                )
                img_qkv = self.img_attn_qkv(img_modulated)
                img_q, img_k, img_v = rearrange(
                    img_qkv, "B L (K H D) -> K B L H D", K=3, H=self.heads_num
                )
                # Apply QK-Norm if needed
                img_q = self.img_attn_q_norm(img_q).to(img_v)
                img_k = self.img_attn_k_norm(img_k).to(img_v)

                # Apply RoPE if needed.
                if freqs_cis is not None:
                    img_qq, img_kk = apply_rotary_emb(img_q, img_k, freqs_cis, head_first=False)
                    if not (img_qq.shape == img_q.shape and img_kk.shape == img_k.shape):
                        raise ValueError(
                            f"Rotary embedding output shape mismatch. "
                            f"img_qq shape: {img_qq.shape}, img_q shape: {img_q.shape}, "
                            f"img_kk shape: {img_kk.shape}, img_k shape: {img_k.shape}"
                        )
                    img_q, img_k = img_qq, img_kk

                # Prepare txt for attention.
                txt_modulated = self.txt_norm1(txt)
                txt_modulated = modulate(
                    txt_modulated, shift=txt_mod1_shift, scale=txt_mod1_scale
                )
                txt_qkv = self.txt_attn_qkv(txt_modulated)
                txt_q, txt_k, txt_v = rearrange(
                    txt_qkv, "B L (K H D) -> K B L H D", K=3, H=self.heads_num
                )
                # Apply QK-Norm if needed.
                txt_q = self.txt_attn_q_norm(txt_q).to(txt_v)
                txt_k = self.txt_attn_k_norm(txt_k).to(txt_v)

                # Run actual attention.
                q = torch.cat((img_q, txt_q), dim=1)
                k = torch.cat((img_k, txt_k), dim=1)
                v = torch.cat((img_v, txt_v), dim=1)
                expected_cu_seqlens_q_length = 2 * img.shape[0] + 1
                if cu_seqlens_q.shape[0] != expected_cu_seqlens_q_length:
                    raise ValueError(
                        f"cu_seqlens_q shape mismatch: "
                        f"cu_seqlens_q.shape:{cu_seqlens_q.shape}, img.shape[0]:{img.shape[0]}"
                        f"expected first dimension length: {expected_cu_seqlens_q_length}"
                    )

                # ===== 应用在线旋转（在 FA3 量化之前）=====
                if hasattr(self, 'q_rot'):
                    q = self.q_rot(q)
                if hasattr(self, 'k_rot'):
                    k = self.k_rot(k)
                # ==========================================

                # ===== 插入 FA3 占位 =====
                if hasattr(self, 'fa3_q'):
                    q = self.fa3_q(q)
                if hasattr(self, 'fa3_k'):
                    k = self.fa3_k(k)
                if hasattr(self, 'fa3_v'):
                    v = self.fa3_v(v)
                # ========================

                # attention computation start
                if not self.hybrid_seq_parallel_attn:
                    attn = attention(
                        q,
                        k,
                        v,
                        mode="torch",
                        cu_seqlens_q=cu_seqlens_q,
                        cu_seqlens_kv=cu_seqlens_kv,
                        max_seqlen_q=max_seqlen_q,
                        max_seqlen_kv=max_seqlen_kv,
                        batch_size=img_k.shape[0],
                    )
                else:
                    attn = parallel_attention(
                        self.hybrid_seq_parallel_attn,
                        q,
                        k,
                        v,
                        img_q_len=img_q.shape[1],
                        img_kv_len=img_k.shape[1],
                        cu_seqlens_q=cu_seqlens_q,
                        cu_seqlens_kv=cu_seqlens_kv
                    )
                
                # attention computation end

                img_attn, txt_attn = attn[:, : img.shape[1]], attn[:, img.shape[1] :]

                # Calculate the img blocks.
                img = img + apply_gate(self.img_attn_proj(img_attn), gate=img_mod1_gate)
                img = img + apply_gate(
                    self.img_mlp(
                        modulate(
                            self.img_norm2(img), shift=img_mod2_shift, scale=img_mod2_scale
                        )
                    ),
                    gate=img_mod2_gate,
                )

                # Calculate the txt blocks.
                txt = txt + apply_gate(self.txt_attn_proj(txt_attn), gate=txt_mod1_gate)
                txt = txt + apply_gate(
                    self.txt_mlp(
                        modulate(
                            self.txt_norm2(txt), shift=txt_mod2_shift, scale=txt_mod2_scale
                        )
                    ),
                    gate=txt_mod2_gate,
                )

                return img, txt

            module.forward = new_forward.__get__(module, module.__class__)

        def _wrap_single_forward(module: nn.Module):
            """包裹 MMSingleStreamBlock 的 forward 方法"""
            original_forward = module.forward
            
            # 动态导入必要的函数
            hyvideo_single_module = import_module(original_forward.__module__)
            modulate = hyvideo_single_module.modulate
            apply_gate = hyvideo_single_module.apply_gate
            rearrange = hyvideo_single_module.rearrange
            apply_rotary_emb = hyvideo_single_module.apply_rotary_emb
            attention = hyvideo_single_module.attention
            parallel_attention = hyvideo_single_module.parallel_attention

            def new_forward(
                    self,
                    x: torch.Tensor,
                    vec: torch.Tensor,
                    txt_len: int,
                    cu_seqlens_q: Optional[torch.Tensor] = None,
                    cu_seqlens_kv: Optional[torch.Tensor] = None,
                    max_seqlen_q: Optional[int] = None,
                    max_seqlen_kv: Optional[int] = None,
                    freqs_cis: Tuple[torch.Tensor, torch.Tensor] = None,
                ) -> torch.Tensor:
                mod_shift, mod_scale, mod_gate = self.modulation(vec).chunk(3, dim=-1)
                x_mod = modulate(self.pre_norm(x), shift=mod_shift, scale=mod_scale)
                qkv, mlp = torch.split(
                    self.linear1(x_mod), [3 * self.hidden_size, self.mlp_hidden_dim], dim=-1
                )
                
                q, k, v = rearrange(qkv, "B L (K H D) -> K B L H D", K=3, H=self.heads_num)

                # Apply QK-Norm if needed.
                q = self.q_norm(q).to(v)
                k = self.k_norm(k).to(v)

                # Apply RoPE if needed.
                if freqs_cis is not None:
                    img_q, txt_q = q[:, :-txt_len, :, :], q[:, -txt_len:, :, :]
                    img_k, txt_k = k[:, :-txt_len, :, :], k[:, -txt_len:, :, :]
                    img_qq, img_kk = apply_rotary_emb(img_q, img_k, freqs_cis, head_first=False)
                    if not (img_qq.shape == img_q.shape and img_kk.shape == img_k.shape):
                        raise ValueError(
                            f"Rotary embedding output shape mismatch. "
                            f"img_qq shape: {img_qq.shape}, img_q shape: {img_q.shape}, "
                            f"img_kk shape: {img_kk.shape}, img_k shape: {img_k.shape}"
                        )
                    img_q, img_k = img_qq, img_kk
                    q = torch.cat((img_q, txt_q), dim=1)
                    k = torch.cat((img_k, txt_k), dim=1)
                else:
                    # 如果 freqs_cis 为 None，需要计算 img_q_len 和 img_kv_len 用于 parallel_attention
                    img_q_len = q.shape[1] - txt_len
                    img_kv_len = k.shape[1] - txt_len

                # Compute attention.
                expected_cu_seqlens_q_length = 2 * x.shape[0] + 1
                if cu_seqlens_q.shape[0] != expected_cu_seqlens_q_length:
                    raise ValueError(
                        f"cu_seqlens_q shape mismatch. "
                        f"cu_seqlens_q.shape: {cu_seqlens_q.shape}, x.shape[0]: {x.shape[0]}, "
                        f"expected first dimension length: {expected_cu_seqlens_q_length}"
                    )

                # ===== 应用在线旋转（在 FA3 量化之前）=====
                if hasattr(self, 'q_rot'):
                    q = self.q_rot(q)
                if hasattr(self, 'k_rot'):
                    k = self.k_rot(k)
                # ==========================================

                # ===== 插入 FA3 占位 =====
                if hasattr(self, 'fa3_q'):
                    q = self.fa3_q(q)
                if hasattr(self, 'fa3_k'):
                    k = self.fa3_k(k)
                if hasattr(self, 'fa3_v'):
                    v = self.fa3_v(v)
                # ========================

                # attention computation start
                if not self.hybrid_seq_parallel_attn:
                    attn = attention(
                        q,
                        k,
                        v,
                        mode="torch",
                        cu_seqlens_q=cu_seqlens_q,
                        cu_seqlens_kv=cu_seqlens_kv,
                        max_seqlen_q=max_seqlen_q,
                        max_seqlen_kv=max_seqlen_kv,
                        batch_size=x.shape[0],
                    )
                else:
                    # 如果 freqs_cis 不为 None，使用 img_q 和 img_k 的长度；否则使用计算出的长度
                    if freqs_cis is not None:
                        img_q_len_val = img_q.shape[1]
                        img_kv_len_val = img_k.shape[1]
                    else:
                        img_q_len_val = img_q_len
                        img_kv_len_val = img_kv_len
                    attn = parallel_attention(
                        self.hybrid_seq_parallel_attn,
                        q,
                        k,
                        v,
                        img_q_len=img_q_len_val,
                        img_kv_len=img_kv_len_val,
                        cu_seqlens_q=cu_seqlens_q,
                        cu_seqlens_kv=cu_seqlens_kv
                    )
                # attention computation end

                # Compute activation in mlp stream, cat again and run second linear layer.
                output = self.linear2(torch.cat((attn, self.mlp_act(mlp)), 2))
                return x + apply_gate(output, gate=mod_gate)

            module.forward = new_forward.__get__(module, module.__class__)

        # 遍历并注入占位符
        for name, module in root_module.named_modules():
            module_type = module.__class__.__name__
            
            # 检查是否是目标模块类型
            if module_type not in ["MMDoubleStreamBlock", "MMSingleStreamBlock"]:
                continue
            
            full_name = f"{root_name}.{name}" if root_name else name
            if not should_inject(full_name):
                continue
            if name == "":
                prefix = ""
            else:
                prefix = f"{name}."
            # 为该模块注入占位符
            root_module.set_submodule(f"{prefix}fa3_q", FA3QuantPlaceHolder(ratio=0.9999))
            root_module.set_submodule(f"{prefix}fa3_k", FA3QuantPlaceHolder(ratio=0.9999))
            root_module.set_submodule(f"{prefix}fa3_v", FA3QuantPlaceHolder(ratio=1.0))
            
            # 包裹对应的 forward 方法
            if module_type == "MMDoubleStreamBlock":
                _wrap_double_forward(module)
            elif module_type == "MMSingleStreamBlock":
                _wrap_single_forward(module)
            
            get_logger().info(f"Injected FA3 placeholders for {full_name}")