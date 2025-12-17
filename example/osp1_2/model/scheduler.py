# Copyright 2024 Katherine Crowson and The HuggingFace Team. All rights reserved.
# Copyright Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.

from typing import Optional, Union, List
import numpy as np
import torch

from diffusers import EulerAncestralDiscreteScheduler as EulerAncestralDiscreteScheduler_base
from diffusers.configuration_utils import register_to_config


# Copied and modified from diffusers.schedulers.scheduling_euler_ancestral_discrete.EulerAncestralDiscreteScheduler
class EulerAncestralDiscreteSchedulerExample(EulerAncestralDiscreteScheduler_base):
    @register_to_config
    def __init__(
            self,
            num_train_timesteps: int = 1000,
            beta_start: float = 0.0001,
            beta_end: float = 0.02,
            beta_schedule: str = "linear",
            trained_betas: Optional[Union[np.ndarray, List[float]]] = None,
            prediction_type: str = "epsilon",
            timestep_spacing: str = "linspace",
            steps_offset: int = 0,
            rescale_betas_zero_snr: bool = False,
    ):
        super().__init__(
            num_train_timesteps=num_train_timesteps,
            beta_start=beta_start,
            beta_end=beta_end,
            beta_schedule=beta_schedule,
            trained_betas=trained_betas,
            prediction_type=prediction_type,
            timestep_spacing=timestep_spacing,
            steps_offset=steps_offset,
            rescale_betas_zero_snr=rescale_betas_zero_snr,
        )

    def sigma(self, t):
        base_sigmas = np.sqrt((1 - self.alphas_cumprod) / self.alphas)
        base_timesteps = np.arange(0, len(base_sigmas))
        return np.interp(t, base_timesteps, base_sigmas).tolist()

    def set_timesteps(
            self,
            num_inference_steps: int = None,
            timesteps: Union[list, torch.Tensor] = None,
            device: Union[str, torch.device] = None
    ):
        num_inference_steps = num_inference_steps if num_inference_steps is not None else len(timesteps)

        # Sets timesteps and converts the input to a NumPy array.
        if timesteps is not None:
            if not isinstance(timesteps, (list, torch.Tensor, np.ndarray)):
                raise ValueError("timesteps must be a list, np.array, or a torch.Tensor when provided.")

            # If timesteps is a torch.Tensor, move to device if provided, then convert to np.array
            if isinstance(timesteps, torch.Tensor):
                if device is not None:
                    timesteps = timesteps.to(device)
                timesteps = timesteps.cpu().numpy()
            # If timesteps is a list, convert it to np.array
            elif isinstance(timesteps, list):
                timesteps = np.array(timesteps)
            # If timesteps is already an np.array, we keep it as is

        else:
            # "linspace", "leading", "trailing" corresponds to annotation of
            #   Table 2. of https://arxiv.org/abs/2305.08891
            if self.config.timestep_spacing == "linspace":
                timesteps = np.linspace(0, self.config.num_train_timesteps - 1, num_inference_steps, dtype=np.float32)[
                            ::-1
                            ].copy()
            elif self.config.timestep_spacing == "leading":
                step_ratio = self.config.num_train_timesteps // self.num_inference_steps
                # creates integer timesteps by multiplying by ratio
                # casting to int to avoid issues when num_inference_step is power of 3
                timesteps = (np.arange(0, num_inference_steps) * step_ratio).round()[::-1].copy().astype(np.float32)
                timesteps += self.config.steps_offset
            elif self.config.timestep_spacing == "trailing":
                step_ratio = self.config.num_train_timesteps / self.num_inference_steps
                # creates integer timesteps by multiplying by ratio
                # casting to int to avoid issues when num_inference_step is power of 3
                timesteps = (np.arange(self.config.num_train_timesteps, 0, -step_ratio)).round().copy().astype(
                    np.float32)
                timesteps -= 1
            else:
                raise ValueError(
                    f"{self.config.timestep_spacing} is not supported. "
                    f"Please make sure to choose one of 'linspace', 'leading' or 'trailing'."
                )

        sigmas = np.array(((1 - self.alphas_cumprod) / self.alphas_cumprod) ** 0.5)
        sigmas = np.interp(timesteps, np.arange(0, len(sigmas)), sigmas)
        sigmas = np.concatenate([sigmas, [0.0]]).astype(np.float32)
        self.sigmas = torch.from_numpy(sigmas).to(device=device)

        self.timesteps = torch.from_numpy(timesteps).to(device=device)
        self._step_index = None
        self._begin_index = None
        self.sigmas = self.sigmas.to("cpu")  # to avoid too much CPU/GPU communication
