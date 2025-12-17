# -*- coding: utf-8 -*-
# Copyright (c) 2024-2024 Huawei Technologies Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
import shutil
from typing import Dict, Union, Optional

import pytest
import torch
from accelerate import dispatch_model
from accelerate.hooks import AlignDevicesHook
from torch import nn
from transformers import PreTrainedModel, LlamaConfig, LlamaForCausalLM

from msmodelslim.pytorch.llm_ptq.accelerate_adapter.hook_adapter import UpdateWeightsMapHook
from msmodelslim.pytorch.llm_ptq.accelerate_adapter.hook_adapter import (replace_device_align_hook_if_needed,
                                                                         move_update_weight_hook_if_need,
                                                                         PrepareWeight)
from msmodelslim.pytorch.llm_ptq.accelerate_adapter.utils import HF_HOOK

ALL_CPU_DEVICE_MAP = {
    'model.embed_tokens': 'cpu',
    'model.norm': 'cpu',
    'model.rotary_emb': 'cpu',
    'lm_head': 'cpu',
    'model.layers': 'cpu',
}

CPU_DISK_DEVICE_MAP = {
    'model.embed_tokens': 'cpu',
    'model.norm': 'cpu',
    'model.rotary_emb': 'cpu',
    'lm_head': 'cpu',
    'model.layers': 'disk',
}

OFFLOAD_DIR = './offload'


@pytest.fixture
def clear_offload_dir():
    yield

    if os.path.exists(OFFLOAD_DIR):
        shutil.rmtree(OFFLOAD_DIR)


def get_fake_dispatched_llama_model(
        device_map: Dict[str, Union[str, int, torch.device]] = None,
        main_device: Optional[torch.device] = None
):
    """
    获取一个随机的、非常小的Llama模型，并将其dispatch至相应的设备
    """
    config_path = os.path.join(os.path.dirname(__file__), "config.json")
    config = LlamaConfig.from_json_file(config_path)
    model = LlamaForCausalLM(config)
    device_map = ALL_CPU_DEVICE_MAP if device_map is None else device_map
    # 1. 保存原始 umask
    original_umask = os.umask(0)  # 临时设为 0 并获取原始值
    try:
        # 2. 设置目标 umask（0o026 对应权限 640）
        os.umask(0o026)
        dispatch_model(
            model,
            device_map=device_map,
            main_device=main_device,
            state_dict=model.state_dict(),
            offload_dir=OFFLOAD_DIR
        )
    finally:
        # 4. 无论是否出错，都恢复原始 umask
        os.umask(original_umask)
    return model


def check_all_hook(model: PreTrainedModel, target_hook_type, allow_none: bool = True):
    for _, module in model.named_modules():

        hook = getattr(module, HF_HOOK, None)

        if allow_none and hook is None:
            continue

        if not isinstance(hook, target_hook_type):
            return False

    return True


def test_no_error_if_model_has_no_device_align_hook(clear_offload_dir):
    model = get_fake_dispatched_llama_model(ALL_CPU_DEVICE_MAP)
    assert not any([value.device.type == 'meta' for _, value in model.state_dict().items()])
    replace_device_align_hook_if_needed(model)
    assert check_all_hook(model, AlignDevicesHook)


def test_replace_result_if_model_has_device_align_hook(clear_offload_dir):
    model = get_fake_dispatched_llama_model(CPU_DISK_DEVICE_MAP)
    assert any([value.device.type == 'meta' for _, value in model.state_dict().items()])
    replace_device_align_hook_if_needed(model)
    assert check_all_hook(model, UpdateWeightsMapHook)


def test_prepare_weight_can_load_weight_from_disk(clear_offload_dir):
    model = get_fake_dispatched_llama_model(CPU_DISK_DEVICE_MAP)
    replace_device_align_hook_if_needed(model)
    assert any([value.device.type == 'meta' for _, value in model.state_dict().items()])

    for _, module in model.named_modules():
        if isinstance(module, nn.Linear):
            with PrepareWeight(module):
                assert not any([value.device.type == 'meta' for _, value in module.state_dict().items()])


@torch.no_grad()
def test_prepare_weight_can_update_weight_with_post_force(clear_offload_dir):
    model = get_fake_dispatched_llama_model(CPU_DISK_DEVICE_MAP)
    replace_device_align_hook_if_needed(model)
    assert any([value.device.type == 'meta' for _, value in model.state_dict().items()])

    for _, module in model.named_modules():
        if isinstance(module, nn.Linear):
            with PrepareWeight(module, post_force=True):
                assert not any([value.device.type == 'meta' for _, value in module.state_dict().items()])
                old_weight = module.weight.clone().detach()
                module.weight.div_(torch.tensor([0.5]))
                updated_weight_value = module.weight.clone().detach()

            with PrepareWeight(module):
                new_load_weight = module.weight.clone().detach()

            assert not torch.equal(old_weight, updated_weight_value)
            assert torch.equal(updated_weight_value, new_load_weight)


@torch.no_grad()
def test_prepare_weight_can_manage_new_submodule_with_post_recurse(clear_offload_dir):
    model = get_fake_dispatched_llama_model(CPU_DISK_DEVICE_MAP)
    replace_device_align_hook_if_needed(model)
    assert any([value.device.type == 'meta' for _, value in model.state_dict().items()])

    original_umask = os.umask(0)
    try:
        os.umask(0o026)
        for _, module in model.named_modules():
            if isinstance(module, nn.Linear):
                with PrepareWeight(module, post_recurse=True):
                    module.sub_module = nn.LayerNorm(128, device=module.weight.device)
                    module.sub_module.weight.data = torch.ones([128, 128], device=module.weight.device)
                    move_update_weight_hook_if_need(module, module, as_submodule=True)

                # new submodule should have same device type with module
                if module.weight.device.type == 'meta':
                    assert module.sub_module.weight.device.type == 'meta'

                # new submodule can be load correctly
                with PrepareWeight(module):
                    assert module.weight.device.type != 'meta'
                    assert module.sub_module.weight.device.type == module.weight.device.type
                    assert torch.equal(torch.ones([128, 128]), module.sub_module.weight)
    finally:
        os.umask(original_umask)

@torch.no_grad()
def test_prepare_weight_can_manage_new_parameters_automatic(clear_offload_dir):
    model = get_fake_dispatched_llama_model(CPU_DISK_DEVICE_MAP)
    replace_device_align_hook_if_needed(model)
    assert any([value.device.type == 'meta' for _, value in model.state_dict().items()])

    original_umask = os.umask(0)
    try:
        os.umask(0o026)
        for _, module in model.named_modules():
            if isinstance(module, nn.Linear):
                with PrepareWeight(module):
                    assert not any([value.device.type == 'meta' for _, value in module.state_dict().items()])
                    module.new_param = nn.Parameter(torch.tensor([1.0]))

                # new param should have same device type with module
                if module.weight.device.type == 'meta':
                    assert module.new_param.device.type == 'meta'

                # new param can be load correctly
                with PrepareWeight(module):
                    assert module.new_param.device.type != 'meta'
                    assert module.new_param.device.type == module.weight.device.type
                    assert torch.equal(torch.tensor([1.0]), module.new_param)
    finally:
        os.umask(original_umask)