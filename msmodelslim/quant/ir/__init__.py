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
    "W16A16sLinear",
    "FakeQuantActivationPerHead",

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

from .attention import FakeQuantDynamicCache
from .activation import FakeQuantActivationPerHead
from .auto import AutoFakeQuantLinear, AutoFakeQuantActivation, AutoFakeQuantDynamicCache
from .const import int8_per_tensor_sym, int8_per_channel_sym, int8_per_token_sym, int8_per_group_sym, \
    int8_per_tensor_asym, int8_per_token_asym, int8_per_channel_asym, int4_per_channel_sym, \
    int8_per_tensor_asym, int8_per_token_asym, int8_per_channel_asym, int4_per_tensor_sym, int4_per_channel_sym, \
    int4_per_channel_asym, int4_per_token_sym, int4_per_group_sym, int4_per_group_asym, int4_per_tensor_asym, \
    int4_per_token_asym, int8_pd_mix_asym, mxfp4_per_block_sym, mxfp8_per_block_sym, \
    fp8_e4m3_per_token_sym, fp8_e4m3_per_tensor_sym, fp8_e4m3_per_channel_sym
from .quarot import QuarotOnlineRotationInfo, QuarotOnlineHeadRotationWrapper, QuarotOnlineKroneckerRotationWrapper, \
    QuarotHeadsRotationHookIR, QuarotKroneckerRotationHookIR
from .w16a16s import W16A16sLinear
from .w4a4_dynamic import W4A4DynamicPerChannelFakeQuantLinear, W4A4DynamicPerGroupFakeQuantLinear
from .w4a8_dynamic import W4A8DynamicFakeQuantLinear
from .w8a8_dynamic import W8A8DynamicPerChannelFakeQuantLinear, W8A8DynamicPerGroupFakeQuantLinear
from .w8a8_fp_dynamic import WFP8AFP8DynamicPerChannelFakeQuantLinear
from .w8a8_pdmix import W8A8PDMixFakeQuantLinear, PDMixState
from .w8a8_static import W8A8StaticFakeQuantLinear
from .w8a8_mx_dynamic import W8A8MXDynamicPerBlockFakeQuantLinear
from .w4a8_mx_dynamic import W4A8MXDynamicPerBlockFakeQuantLinear
from .w4a4_mx_dynamic import W4A4MXDynamicPerBlockFakeQuantLinear
from .wrapper import WrapperIR, HookIR
