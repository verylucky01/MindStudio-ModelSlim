# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
from enum import Enum


class DeviceType(str, Enum):
    NPU = "npu"  # 昇腾NPU
    CPU = "cpu"  # CPU


class QuantType(str, Enum):
    W4A8 = "w4a8"  # 权重INT4量化，激活值INT8量化
    W4A8C8 = "w4a8c8"  # 权重INT4量化，激活值INT8量化，KVCache INT8量化
    W8A16 = "w8a16"  # 权重INT8量化，激活值不量化
    W8A8 = "w8a8"  # 权重INT8量化，激活值INT8量化
    W8A8S = "w8a8s"  # 权重INT8稀疏量化，激活值INT8量化
    W8A8C8 = "w8a8c8"  # 权重INT8量化，激活值INT8量化，KVCache INT8量化
    W16A16S = "w16a16s"  # 权重浮点稀疏


class RunnerType(str, Enum):
    """Runner类型枚举"""
    AUTO = "auto"
    MODEL_WISE = "model_wise"
    LAYER_WISE = "layer_wise"
    DP_LAYER_WISE = "dp_layer_wise"
