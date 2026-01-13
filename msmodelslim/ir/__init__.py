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

__all__ = [
    "WrapperIR",
    "HookIR",
    "AutoFakeQuantLinear",
    "AutoFakeQuantActivation",
    "AutoFakeQuantDynamicCache",
    "W8A8StaticFakeQuantLinear",
    "W8A8DynamicPerChannelFakeQuantLinear",
    "W8A8MXDynamicPerBlockFakeQuantLinear",
    "W4A4MXDynamicPerBlockFakeQuantLinear",
    "W4A8MXDynamicPerBlockFakeQuantLinear",
    "W8A8PDMixFakeQuantLinear",
    "W8A8DynamicPerGroupFakeQuantLinear",
    "W4A4DynamicPerChannelFakeQuantLinear",
    "W4A4DynamicPerGroupFakeQuantLinear",
    "W4A8DynamicFakeQuantLinear",
    "WFP8AFP8DynamicPerChannelFakeQuantLinear",
    "FakeQuantDynamicCache",
    "QuarotOnlineRotationInfo",
    "QuarotOnlineHeadRotationWrapper",
    "QuarotOnlineKroneckerRotationWrapper",
    "QuarotHeadsRotationHookIR",
    "QuarotKroneckerRotationHookIR",
    "OnlineRotationInfo",
    "OnlineRotationWrapper",
    "OnlineRotationInputHookIR",
    "OnlineRotationOutputHookIR",
    "W16A16sLinear",
    "FakeQuantActivationPerHead",
    "FakeQuantActivationPerToken",

    "int8_per_tensor_sym",
    "int8_per_channel_sym",
    "int8_per_channel_asym",
    "int8_per_token_sym",
    "int8_per_group_sym",
    "int8_per_tensor_asym",
    "int8_per_token_asym",
    "int8_pd_mix_asym",

    "int4_per_tensor_sym",
    "int4_per_channel_sym",
    "int4_per_channel_asym",
    "int4_per_token_sym",
    "int4_per_group_sym",
    "int4_per_group_asym",
    "int4_per_tensor_asym",
    "int4_per_token_asym",
    "mxfp4_per_block_sym",
    "mxfp8_per_block_sym",
    "PDMixState",
]

from .activation import FakeQuantActivationPerHead
from .activation_dynamic import FakeQuantActivationPerToken
from .api.api_main import *
from .attention import FakeQuantDynamicCache
from .auto import AutoFakeQuantLinear, AutoFakeQuantActivation, AutoFakeQuantDynamicCache
from .const import int8_per_tensor_sym, int8_per_channel_sym, int8_per_token_sym, int8_per_group_sym, \
    int8_per_tensor_asym, int8_per_token_asym, int8_per_channel_asym, int4_per_channel_sym, \
    int8_per_tensor_asym, int8_per_token_asym, int8_per_channel_asym, int4_per_tensor_sym, int4_per_channel_sym, \
    int4_per_channel_asym, int4_per_token_sym, int4_per_group_sym, int4_per_group_asym, int4_per_tensor_asym, \
    int4_per_token_asym, int8_pd_mix_asym, mxfp4_per_block_sym, mxfp8_per_block_sym, \
    fp8_e4m3_per_token_sym, fp8_e4m3_per_tensor_sym, fp8_e4m3_per_channel_sym, \
    int8_per_head_sym
from .quarot import QuarotOnlineRotationInfo, QuarotOnlineHeadRotationWrapper, QuarotOnlineKroneckerRotationWrapper, \
    QuarotHeadsRotationHookIR, QuarotKroneckerRotationHookIR, OnlineRotationInfo, OnlineRotationWrapper, \
    OnlineRotationInputHookIR, OnlineRotationOutputHookIR
from .w16a16s import W16A16sLinear
from .w4a4_dynamic import W4A4DynamicPerChannelFakeQuantLinear, W4A4DynamicPerGroupFakeQuantLinear
from .w4a4_mx_dynamic import W4A4MXDynamicPerBlockFakeQuantLinear
from .w4a8_dynamic import W4A8DynamicFakeQuantLinear
from .w4a8_mx_dynamic import W4A8MXDynamicPerBlockFakeQuantLinear
from .w8a8_dynamic import W8A8DynamicPerChannelFakeQuantLinear, W8A8DynamicPerGroupFakeQuantLinear
from .w8a8_fp_dynamic import WFP8AFP8DynamicPerChannelFakeQuantLinear
from .w8a8_mx_dynamic import W8A8MXDynamicPerBlockFakeQuantLinear
from .w8a8_pdmix import W8A8PDMixFakeQuantLinear, PDMixState
from .w8a8_static import W8A8StaticFakeQuantLinear
from .wrapper import WrapperIR, HookIR
