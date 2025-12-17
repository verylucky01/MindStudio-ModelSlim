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

from msmodelslim.core.QAL import QDType
from msmodelslim.core.QAL.qbase import QScheme, QScope

int8_per_token_asym = QScheme(scope=QScope.PER_TOKEN, dtype=QDType.INT8, symmetric=False)
int8_per_token_sym = QScheme(scope=QScope.PER_TOKEN, dtype=QDType.INT8, symmetric=True)
int8_per_tensor_asym = QScheme(scope=QScope.PER_TENSOR, dtype=QDType.INT8, symmetric=False)
int8_per_tensor_sym = QScheme(scope=QScope.PER_TENSOR, dtype=QDType.INT8, symmetric=True)
int8_per_channel_asym = QScheme(scope=QScope.PER_CHANNEL, dtype=QDType.INT8, symmetric=False)
int8_per_channel_sym = QScheme(scope=QScope.PER_CHANNEL, dtype=QDType.INT8, symmetric=True)
int8_per_group_sym = QScheme(scope=QScope.PER_GROUP, dtype=QDType.INT8, symmetric=True)
int8_per_group_asym = QScheme(scope=QScope.PER_GROUP, dtype=QDType.INT8, symmetric=False)
int8_pd_mix_asym = QScheme(scope=QScope.PD_MIX, dtype=QDType.INT8, symmetric=False)
mxfp4_per_block_sym = QScheme(scope=QScope.PER_BLOCK, dtype=QDType.MXFP4, symmetric=True)
mxfp8_per_block_sym = QScheme(scope=QScope.PER_BLOCK, dtype=QDType.MXFP8, symmetric=True)
int8_per_head_sym = QScheme(scope=QScope.PER_HEAD, dtype=QDType.INT8, symmetric=True)

int4_per_token_asym = QScheme(scope=QScope.PER_TOKEN, dtype=QDType.INT4, symmetric=False)
int4_per_token_sym = QScheme(scope=QScope.PER_TOKEN, dtype=QDType.INT4, symmetric=True)
int4_per_tensor_asym = QScheme(scope=QScope.PER_TENSOR, dtype=QDType.INT4, symmetric=False)
int4_per_tensor_sym = QScheme(scope=QScope.PER_TENSOR, dtype=QDType.INT4, symmetric=True)
int4_per_channel_asym = QScheme(scope=QScope.PER_CHANNEL, dtype=QDType.INT4, symmetric=False)
int4_per_channel_sym = QScheme(scope=QScope.PER_CHANNEL, dtype=QDType.INT4, symmetric=True)
int4_per_group_asym = QScheme(scope=QScope.PER_GROUP, dtype=QDType.INT4, symmetric=False)
int4_per_group_sym = QScheme(scope=QScope.PER_GROUP, dtype=QDType.INT4, symmetric=True)

fp8_e4m3_per_token_sym = QScheme(scope=QScope.PER_TOKEN, dtype=QDType.FP8_E4M3, symmetric=True)
fp8_e4m3_per_tensor_sym = QScheme(scope=QScope.PER_TENSOR, dtype=QDType.FP8_E4M3, symmetric=True)
fp8_e4m3_per_channel_sym = QScheme(scope=QScope.PER_CHANNEL, dtype=QDType.FP8_E4M3, symmetric=True)
