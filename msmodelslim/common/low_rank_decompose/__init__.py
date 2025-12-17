# Copyright Huawei Technologies Co., Ltd. 2022-2022. All rights reserved.

from msmodelslim.common.low_rank_decompose.low_rank_decompose import (
    RankMethods,
    get_hidden_channels_by_layer_name,
    is_hidden_channels_valid,
    get_decompose_channels_2d,
    decompose_weight_2d_svd,
    get_decompose_channels_4d,
    decompose_weight_4d_tucker,
)
