# Copyright 2024 PixArt-Sigma Authors and The HuggingFace Team. All rights reserved.
# Copyright Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.

from typing import Union, Optional, List, Tuple, Callable

import torch.distributed

from diffusers import AutoencoderKL, DPMSolverMultistepScheduler, Transformer2DModel
from einops import rearrange
from transformers import T5EncoderModel

try:
    from opensora.sample.pipeline_opensora_sp import OpenSoraPipeline, T5Tokenizer, hccl_info, \
        get_sequence_parallel_state, ImagePipelineOutput, retrieve_timesteps, torch_npu

except ImportError as e:
    raise ImportError("Cannot find package Open-Sora-Plan 1.2, please install it first.") from e
NCCL_INFO = None


class OneStepSampleArgs:
    def __init__(
            self,
            latents,
            timestep,
            step_index,
            encoder_states,
            extra_step_kwargs,
            added_cond_kwargs
    ) -> None:
        self.latents = latents
        self.timestep = timestep
        self.step_index = step_index
        self.encoder_states = encoder_states
        self.extra_step_kwargs = extra_step_kwargs
        self.added_cond_kwargs = added_cond_kwargs


class TextEmbeddingsArgs:
    def __init__(
            self,
            prompt: Union[str, List[str]] = None,
            negative_prompt: str = "",
            guidance_scale: float = 4.5,
            num_images_per_prompt: Optional[int] = 1,
            prompt_embeds: Optional[torch.FloatTensor] = None,
            prompt_attention_mask: Optional[torch.FloatTensor] = None,
            negative_prompt_embeds: Optional[torch.FloatTensor] = None,
            negative_prompt_attention_mask: Optional[torch.FloatTensor] = None,
            clean_caption: bool = True,
            max_sequence_length: int = 300,
            **kwargs
    ) -> None:
        self.prompt = prompt
        self.negative_prompt = negative_prompt
        self.guidance_scale = guidance_scale
        self.num_images_per_prompt = num_images_per_prompt
        self.prompt_embeds = prompt_embeds
        self.prompt_attention_mask = prompt_attention_mask
        self.negative_prompt_embeds = negative_prompt_embeds
        self.negative_prompt_attention_mask = negative_prompt_attention_mask
        self.clean_caption = clean_caption
        self.max_sequence_length = max_sequence_length
        self.kwargs = kwargs


# Copy and modified from Open-Sora-Plan repo v1.2: opensora.sample.pipeline_opensora_sp
class OpenSoraPipelineV1x2(OpenSoraPipeline):
    def __init__(self, tokenizer: T5Tokenizer, text_encoder: T5EncoderModel, vae: AutoencoderKL,
                 transformer: Transformer2DModel, scheduler: DPMSolverMultistepScheduler):
        super().__init__(tokenizer, text_encoder, vae, transformer, scheduler)
        self.args = None

    @torch.no_grad()
    def __call__(
            self,
            prompt: Union[str, List[str]] = None,
            negative_prompt: str = "",
            num_inference_steps: int = 20,
            timesteps: List[int] = None,
            guidance_scale: float = 4.5,
            num_images_per_prompt: Optional[int] = 1,
            num_frames: Optional[int] = None,
            height: Optional[int] = None,
            width: Optional[int] = None,
            eta: float = 0.0,
            generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
            latents: Optional[torch.FloatTensor] = None,
            prompt_embeds: Optional[torch.FloatTensor] = None,
            prompt_attention_mask: Optional[torch.FloatTensor] = None,
            negative_prompt_embeds: Optional[torch.FloatTensor] = None,
            negative_prompt_attention_mask: Optional[torch.FloatTensor] = None,
            output_type: Optional[str] = "pil",
            return_dict: bool = True,
            callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
            callback_steps: int = 1,
            clean_caption: bool = True,
            use_resolution_binning: bool = True,
            max_sequence_length: int = 300,
            **kwargs,
    ) -> Union[ImagePipelineOutput, Tuple]:
        """
        (Deprecated) Copy and modified from `super().__call__` to add `DitCacheAdaptor.set_timestep_idx(i)`
        (The newer version of DitCacheAdaptor no longer need this.
        It only needs to set time step before the initial run.)
        """

        # 1. Check inputs. Raise error if not correct
        num_frames = num_frames or self.transformer.config.sample_size_t * self.vae.vae_scale_factor[0]
        height = height or self.transformer.config.sample_size[0] * self.vae.vae_scale_factor[1]
        width = width or self.transformer.config.sample_size[1] * self.vae.vae_scale_factor[2]
        self.check_inputs(
            prompt,
            num_frames,
            height,
            width,
            negative_prompt,
            callback_steps,
            prompt_embeds,
            negative_prompt_embeds,
            prompt_attention_mask,
            negative_prompt_attention_mask,
        )

        # 2. Default height and width to transformer
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]
        device = getattr(self, '_execution_device', None) or getattr(self, 'device', None) or torch.device('cuda')
        device = kwargs.get('device', device)  # fix bug

        do_classifier_free_guidance = guidance_scale > 1.0

        # 3. Encode input prompt
        (
            prompt_embeds,
            prompt_attention_mask,
            negative_prompt_embeds,
            negative_prompt_attention_mask,
        ) = self.encode_prompt(
            prompt,
            do_classifier_free_guidance,
            negative_prompt=negative_prompt,
            num_images_per_prompt=num_images_per_prompt,
            device=device,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            prompt_attention_mask=prompt_attention_mask,
            negative_prompt_attention_mask=negative_prompt_attention_mask,
            clean_caption=clean_caption,
            max_sequence_length=max_sequence_length,
        )
        if do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
            prompt_attention_mask = torch.cat([negative_prompt_attention_mask, prompt_attention_mask], dim=0)

        # 4. Prepare timesteps
        timesteps, num_inference_steps = retrieve_timesteps(self.scheduler, num_inference_steps, device, timesteps)

        # 5. Prepare latents.
        latent_channels = self.transformer.config.in_channels
        world_size = hccl_info.world_size if torch_npu is not None else NCCL_INFO.world_size
        if world_size == 0:
            raise ZeroDivisionError("world_size can not be zero")
        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            latent_channels,
            (num_frames + world_size - 1) // world_size if get_sequence_parallel_state() else num_frames,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
        )
        if get_sequence_parallel_state():
            prompt_embeds = rearrange(prompt_embeds, 'b (n x) h -> b n x h', n=world_size,
                                      x=prompt_embeds.shape[1] // world_size).contiguous()
            rank = hccl_info.rank if torch_npu is not None else NCCL_INFO.rank
            prompt_embeds = prompt_embeds[:, rank, :, :]

        # 6. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # 6.1 Prepare micro-conditions.
        added_cond_kwargs = {"resolution": None, "aspect_ratio": None}

        # 7. Denoising loop
        num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)

        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                current_timestep = t
                if not torch.is_tensor(current_timestep):
                    is_mps = latent_model_input.device.type == "mps"
                    if isinstance(current_timestep, float):
                        dtype = torch.float32 if is_mps else torch.float64
                    else:
                        dtype = torch.int32 if is_mps else torch.int64
                    current_timestep = torch.tensor([current_timestep], dtype=dtype, device=latent_model_input.device)
                elif len(current_timestep.shape) == 0:
                    current_timestep = current_timestep[None].to(latent_model_input.device)
                # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
                current_timestep = current_timestep.expand(latent_model_input.shape[0])

                if prompt_embeds.ndim == 3:
                    prompt_embeds = prompt_embeds.unsqueeze(1)  # b l d -> b 1 l d
                if prompt_attention_mask.ndim == 2:
                    prompt_attention_mask = prompt_attention_mask.unsqueeze(1)  # b l -> b 1 l
                # prepare attention_mask.
                # b c t h w -> b t h w
                attention_mask = torch.ones_like(latent_model_input)[:, 0]
                if get_sequence_parallel_state():
                    attention_mask = attention_mask.repeat(1, world_size, 1, 1)
                # predict noise model_output
                noise_pred = self.transformer(
                    latent_model_input,
                    attention_mask=attention_mask,
                    encoder_hidden_states=prompt_embeds,
                    encoder_attention_mask=prompt_attention_mask,
                    timestep=current_timestep,
                    added_cond_kwargs=added_cond_kwargs,
                    return_dict=False,
                )[0]

                # perform guidance
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                # learned sigma
                if self.transformer.config.out_channels // 2 == latent_channels:
                    noise_pred = noise_pred.chunk(2, dim=1)[0]
                else:
                    noise_pred = noise_pred

                # compute previous image: x_t -> x_t-1
                latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=False)[0]
                # call the callback, if provided
                if callback_steps == 0 or self.scheduler.order == 0:
                    raise ZeroDivisionError("callback_steps or scheduler order attr cannot be zero.")
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        step_idx = i // getattr(self.scheduler, "order", 1)
                        callback(step_idx, t, latents)
        world_size = hccl_info.world_size if torch_npu is not None else NCCL_INFO.world_size
        if get_sequence_parallel_state():
            latents_shape = list(latents.shape)
            full_shape = [latents_shape[0] * world_size] + latents_shape[1:]
            all_latents = torch.zeros(full_shape, dtype=latents.dtype, device=latents.device)
            torch.distributed.all_gather_into_tensor(all_latents, latents)
            latents_list = list(all_latents.chunk(world_size, dim=0))
            latents = torch.cat(latents_list, dim=2)

        if not output_type == "latent":
            # b t h w c
            image = self.decode_latents(latents)
            image = image[:, :num_frames, :height, :width]
        else:
            image = latents

        # Offload all models
        self.maybe_free_model_hooks()

        if not return_dict:
            return (image,)

        return ImagePipelineOutput(images=image)

    @staticmethod
    def split_sequence(sequence, local_rank, world_size):
        old_shape = sequence.shape
        try:
            x = sequence.shape[2] // world_size
        except ZeroDivisionError as ex:
            logging.error('world_size can not be zero. %s', str(ex))
            raise ex
        sequence = rearrange(sequence.view((*old_shape[:3], -1)), 'b c (n x) s -> b c n x s', n=world_size,
                             x=x).contiguous()
        sequence = sequence[:, :, local_rank, :, :]
        return sequence.view((*old_shape[:2], x, *old_shape[3:]))

    @staticmethod
    def gather_sequences(sequence, world_size):
        sequence_shape = list(sequence.shape)
        full_shape = [sequence_shape[0] * world_size] + sequence_shape[1:]
        all_sequences = torch.zeros(full_shape, dtype=sequence.dtype, device=sequence.device)
        torch.distributed.all_gather_into_tensor(all_sequences, sequence)
        sequences_list = list(all_sequences.chunk(world_size, dim=0))
        sequence = torch.cat(sequences_list, dim=2)
        return sequence

    @staticmethod
    def get_sequence_parallel_state():
        return get_sequence_parallel_state()

    @torch.no_grad()
    def get_text_embeddings(
            self,
            args: TextEmbeddingsArgs
    ) -> Union[ImagePipelineOutput, Tuple]:
        """获取文本嵌入

        Args:
            args: TextEmbeddingsArgs，包含所有获取文本嵌入的相关参数
        """
        # 使用args的命名字段访问参数
        prompt = args.prompt
        negative_prompt = args.negative_prompt
        guidance_scale = args.guidance_scale
        num_images_per_prompt = args.num_images_per_prompt
        prompt_embeds = args.prompt_embeds
        prompt_attention_mask = args.prompt_attention_mask
        negative_prompt_embeds = args.negative_prompt_embeds
        negative_prompt_attention_mask = args.negative_prompt_attention_mask
        clean_caption = args.clean_caption
        max_sequence_length = args.max_sequence_length
        kwargs = args.kwargs

        device = getattr(self, '_execution_device', None) or getattr(self, 'device', None) or torch.device('cuda')
        device = kwargs.get('device', device)  # fix bug

        do_classifier_free_guidance = guidance_scale > 1.0

        # 3. Encode input prompt
        (
            prompt_embeds,
            prompt_attention_mask,
            negative_prompt_embeds,
            negative_prompt_attention_mask,
        ) = self.encode_prompt(
            prompt,
            do_classifier_free_guidance,
            negative_prompt=negative_prompt,
            num_images_per_prompt=num_images_per_prompt,
            device=device,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            prompt_attention_mask=prompt_attention_mask,
            negative_prompt_attention_mask=negative_prompt_attention_mask,
            clean_caption=clean_caption,
            max_sequence_length=max_sequence_length,
        )

        return (
            prompt_embeds,
            prompt_attention_mask,
            negative_prompt_embeds,
            negative_prompt_attention_mask,
        )

    def one_step_sample(self, args: OneStepSampleArgs):
        """执行一步采样

        Args:
            args: OneStepSampleArgs，包含:
                latents: 潜在变量
                timestep: 当前时间步
                step_index: 步骤索引
                encoder_states: 编码器状态
                extra_step_kwargs: 额外步骤参数
                added_cond_kwargs: 额外条件参数
        """
        latents = args.latents
        timestep = args.timestep
        step_index = args.step_index
        encoder_states = args.encoder_states
        extra_step_kwargs = args.extra_step_kwargs
        added_cond_kwargs = args.added_cond_kwargs

        timestep *= self.scheduler.num_train_timesteps

        timesteps_old = self.scheduler.timesteps.cpu().numpy().tolist()
        timesteps = timesteps_old.copy()
        timesteps[step_index] = timestep

        self.scheduler.set_timesteps(len(timesteps), timesteps, device=latents.device)
        latents = self.scheduler.scale_model_input(latents, timestep)

        bs = latents.shape[0]
        current_timestep = torch.tensor([timestep] * bs).to(latents.device)

        if get_sequence_parallel_state():
            latents = self.split_sequence(latents, hccl_info.rank, hccl_info.world_size)

        attention_mask = torch.ones_like(latents)[:, 0]
        if get_sequence_parallel_state():
            attention_mask = attention_mask.repeat(1, hccl_info.world_size, 1, 1)

        noise_pred = self.transformer(
            latents,
            attention_mask=attention_mask,
            timestep=current_timestep,
            added_cond_kwargs=added_cond_kwargs,
            return_dict=False,
            **encoder_states
        )[0]

        latent_channels = self.transformer.config.in_channels
        if self.transformer.config.out_channels // 2 == latent_channels:
            noise_pred = noise_pred.chunk(2, dim=1)[0]
        else:
            noise_pred = noise_pred

        # compute previous image: x_t -> x_t-1
        latents = self.scheduler.step(noise_pred, timestep, latents, **extra_step_kwargs, return_dict=False)[0]
        self.scheduler.set_timesteps(len(timesteps), timesteps_old, device=latents.device)

        if get_sequence_parallel_state():
            latents = self.gather_sequences(latents, hccl_info.world_size)
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        torch.distributed.barrier()
        return latents
