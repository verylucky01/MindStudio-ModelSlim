# Copyright Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
__all__ = ['get_trainable_parameters',
           'FlatQuantQuantizerConfig',
           'quantize_model']


from .flat_quant import (   
    get_trainable_parameters,
    FlatQuantQuantizerConfig,
    quantize_model
)

