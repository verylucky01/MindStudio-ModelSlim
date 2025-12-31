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

import pytest
import torch
from pydantic import ValidationError

from msmodelslim.ir.api import calculate_qparam
from msmodelslim.ir.qal.qbase import QStorage, QDType, QScheme, QScope
from msmodelslim import ir as qir
from msmodelslim.core.observer import MsMinMaxObserver
from msmodelslim.core.quantizer.base import QConfig
from msmodelslim.core.quantizer.impl.ssz import (
    WeightPerChannelSsz
)
from msmodelslim.utils.exception import SpecError


def to_qconfig(q_scheme: QScheme, method: str) -> QConfig:
    q_config = QConfig(
        dtype=q_scheme.dtype.value,
        scope=q_scheme.scope.value,
        symmetric=q_scheme.symmetric,
        method=method,
    )

    if q_scheme.scope == QScope.PER_GROUP:
        q_config.ext['group_size'] = 256

    return q_config


class TestWeightPerChannelSsz:
    """测试Per-Channel ssz量化器"""

    def setup_class(self):
        self.config = QConfig(
            dtype="int8",
            scope="per_channel",
            method="ssz",
            symmetric=True
        )

    def test_initialization(self):
        """测试初始化"""
        quantizer = WeightPerChannelSsz(self.config)

        assert quantizer.config == self.config, \
            "config is not correct, expected: %s, actual: %s" % (self.config, quantizer.config)
        assert quantizer.minmax_observer is not None, \
            "minmax_observer is not correct, expected: %s, actual: %s" % (MsMinMaxObserver, quantizer.minmax_observer)
        assert isinstance(quantizer.minmax_observer, MsMinMaxObserver), \
            "minmax_observer is not correct, expected: %s, actual: %s" % (MsMinMaxObserver, type(quantizer.minmax_observer))
        assert quantizer.weight is None, \
            "weight is not correct, expected: %s, actual: %s" % (None, quantizer.weight)
        assert quantizer.bias is None, \
            "bias is not correct, expected: %s, actual: %s" % (None, quantizer.bias)
        assert quantizer.w_q_param is None, \
            "w_q_param is not correct, expected: %s, actual: %s" % (None, quantizer.w_q_param)
        assert quantizer.w_q_storage is None, \
            "w_q_storage is not correct, expected: %s, actual: %s" % (None, quantizer.w_q_storage)
        assert quantizer.is_quantized is False, \
            "is_quantized is not correct, expected: %s, actual: %s" % (False, quantizer.is_quantized)

    def test_init_weight_validation(self):
        """测试权重初始化验证"""
        quantizer = WeightPerChannelSsz(self.config)

        # 测试无效权重类型
        with pytest.raises(ValidationError, match="instance of QStorage"):
            quantizer.init_weight(torch.randn(10, 20))

        # 测试无效bias类型
        weight = QStorage(QDType.FLOAT, torch.randn(10, 20))
        with pytest.raises(ValidationError, match="instance of Tensor"):
            quantizer.init_weight(weight, bias="invalid")

    def test_get_q_storage_and_q_param_after_forward(self):
        """测试在forward之后获取q_storage和q_param"""
        quantizer = WeightPerChannelSsz(self.config)
        weight = QStorage(QDType.FLOAT, torch.randn(10, 20))
        bias = torch.randn(20)
        quantizer.init_weight(weight, bias)
        quantizer()
        q_storage = quantizer.get_q_storage()
        q_param = quantizer.get_q_param()
        assert q_storage is not None, \
            "q_storage is not correct, expected: %s, actual: %s" % (None, q_storage)
        assert q_param is not None, \
            "q_param is not correct, expected: %s, actual: %s" % (None, q_param)

    def test_forward_before_init_weight(self):
        """测试在初始化权重之前前向传播"""
        quantizer = WeightPerChannelSsz(self.config)

        with pytest.raises(SpecError, match="No weight was set"):
            quantizer()

    def test_forward_after_init_weight(self):
        """测试权重初始化并前向传播"""
        quantizer = WeightPerChannelSsz(self.config)

        # 初始化权重
        weight = QStorage(QDType.FLOAT, torch.randn(10, 20))
        bias = torch.randn(20)

        quantizer.init_weight(weight, bias)

        assert quantizer.weight == weight, \
            "weight is not correct, expected: %s, actual: %s" % (weight, quantizer.weight)
        assert quantizer.bias is bias, \
            "bias is not correct, expected: %s, actual: %s" % (bias, quantizer.bias)

        # 前向传播
        result = quantizer()

        # 验证q_param被设置
        q_param = quantizer.get_q_param()
        assert q_param is not None, \
            "q_param is not correct, expected: %s, actual: %s" % (None, q_param)
        assert q_param.scheme == self.config.to_scheme(), \
            "q_param.scheme is not correct, expected: %s, actual: %s" % (self.config.to_scheme(), q_param.scheme)
        assert isinstance(q_param.ext, dict), \
            "q_param.ext is not correct, expected: %s, actual: %s" % (dict, type(q_param.ext))
        assert "scale" in q_param.ext, \
            "q_param.ext is not correct, expected: %s, actual: %s" % ("scale", q_param.ext.keys())
        assert "offset" in q_param.ext, \
            "q_param.ext is not correct, expected: %s, actual: %s" % ("offset", q_param.ext.keys())
        assert isinstance(q_param.ext["scale"], torch.Tensor), \
            "q_param.ext['scale'] is not correct, expected: %s, actual: %s" % (torch.Tensor, type(q_param.ext["scale"]))
        assert isinstance(q_param.ext["offset"], torch.Tensor), \
            "q_param.ext['offset'] is not correct, expected: %s, actual: %s" % (torch.Tensor, type(q_param.ext["offset"]))
        # Per-channel的scale和offset应该与输出通道数匹配
        assert q_param.ext["scale"].shape == (weight.value.shape[0],), \
            "q_param.ext['scale'].shape is not correct, expected: %s, actual: %s" % ((weight.value.shape[0],), q_param.ext["scale"].shape)
        assert q_param.ext["offset"].shape == (weight.value.shape[0],), \
            "q_param.ext['offset'].shape is not correct, expected: %s, actual: %s" % ((weight.value.shape[0],), q_param.ext["offset"].shape)

        # 验证q_storage被设置
        q_storage = quantizer.get_q_storage()
        assert q_storage is not None, \
            "q_storage is not correct, expected: %s, actual: %s" % (None, q_storage)

        # 验证输出形状
        assert result.shape == weight.value.shape, \
            "result.shape is not correct, expected: %s, actual: %s" % (weight.value.shape, result.shape)

    def test_forward_with_invalid_one_dim_weight(self):
        """测试无效权重形状"""
        quantizer = WeightPerChannelSsz(self.config)
        weight = QStorage(QDType.FLOAT, torch.randn(10))
        bias = torch.randn(10)
        quantizer.init_weight(weight, bias)
        with pytest.raises(SpecError, match="Weight must be a 2D tensor"):
            quantizer()

    def test_forward_with_invalid_three_dim_weight(self):
        """测试无效权重形状"""
        quantizer = WeightPerChannelSsz(self.config)
        weight = QStorage(QDType.FLOAT, torch.randn(10, 20, 30))
        bias = torch.randn(20)
        quantizer.init_weight(weight, bias)
        with pytest.raises(SpecError, match="Weight must be a 2D tensor"):
            quantizer()

    def test_different_weight_shapes(self):
        """测试不同权重形状的处理"""

        # 测试不同形状的权重
        weight_shapes = [(10, 20), (32, 64), (128, 256)]

        for shape in weight_shapes:
            quantizer = WeightPerChannelSsz(self.config)
            weight = QStorage(QDType.FLOAT, torch.randn(*shape))
            bias = torch.randn(shape[1])

            quantizer.init_weight(weight, bias)
            result = quantizer()
            q_param = quantizer.get_q_param()

            assert result.shape == weight.value.shape, \
                "result.shape is not correct, expected: %s, actual: %s" % (weight.value.shape, result.shape)
            assert q_param is not None, \
                "q_param is not correct, expected: %s, actual: %s" % (None, q_param)
            assert q_param.scheme == self.config.to_scheme(), \
                "q_param.scheme is not correct, expected: %s, actual: %s" % (self.config.to_scheme(), q_param.scheme)
            assert q_param.ext["scale"].shape == (shape[0],), \
                "q_param.ext['scale'].shape is not correct, expected: %s, actual: %s" % ((shape[0],), q_param.ext["scale"].shape)
            assert q_param.ext["offset"].shape == (shape[0],), \
                "q_param.ext['offset'].shape is not correct, expected: %s, actual: %s" % ((shape[0],), q_param.ext["offset"].shape)

    @pytest.mark.parametrize(
        "qconfig",
        [
            to_qconfig(qir.int8_per_channel_sym, "ssz"),
        ]
    )
    def test_creation_with_auto_quantizer(self, qconfig):
        """测试通过自动量化器创建"""
        from msmodelslim.core.quantizer.base import AutoWeightQuantizer
        quantizer = AutoWeightQuantizer.from_config(qconfig)
        assert isinstance(quantizer, WeightPerChannelSsz)


class TestSszCalculateQparam:
    """测试 ssz_calculate_qparam 函数"""

    def setup_class(self):
        """设置测试环境"""
        self.symmetric_config = QConfig(
            dtype="int8",
            scope="per_channel",
            method="ssz",
            symmetric=True
        )
        self.asymmetric_config = QConfig(
            dtype="int8",
            scope="per_channel",
            method="ssz",
            symmetric=False
        )

    def test_symmetric_quantization(self):
        """测试对称量化"""
        from msmodelslim.core.quantizer.impl.ssz import ssz_calculate_qparam
        
        # 创建测试权重
        weight = QStorage(QDType.FLOAT, torch.randn(10, 20))
        
        # 创建初始量化参数
        initial_q_param = calculate_qparam(
            min_val=torch.min(weight.T.value, dim=0)[0],
            max_val=torch.max(weight.T.value, dim=0)[0],
            q_dtype=QDType(self.symmetric_config.dtype),
            q_scope=QScope(self.symmetric_config.scope),
            symmetric=self.symmetric_config.symmetric,
        )
        
        # 调用ssz_calculate_qparam
        result_q_param = ssz_calculate_qparam(weight.T, initial_q_param)
        
        # 验证返回的q_param
        assert result_q_param is not None, \
            "q_param is not correct, expected: %s, actual: %s" % (None, result_q_param)
        assert result_q_param.scheme == self.symmetric_config.to_scheme(), \
            "q_param.scheme is not correct, expected: %s, actual: %s" % (self.symmetric_config.to_scheme(), result_q_param.scheme)
        assert "scale" in result_q_param.ext, \
            "q_param.ext is not correct, expected: %s, actual: %s" % ("scale", result_q_param.ext.keys())
        assert "offset" in result_q_param.ext, \
            "q_param.ext is not correct, expected: %s, actual: %s" % ("offset", result_q_param.ext.keys())
        assert isinstance(result_q_param.ext["scale"], torch.Tensor), \
            "q_param.ext['scale'] is not correct, expected: %s, actual: %s" % (torch.Tensor, type(result_q_param.ext["scale"]))
        assert isinstance(result_q_param.ext["offset"], torch.Tensor), \
            "q_param.ext['offset'] is not correct, expected: %s, actual: %s" % (torch.Tensor, type(result_q_param.ext["offset"]))
        assert result_q_param.ext["scale"].shape == (weight.value.shape[0],), \
            "q_param.ext['scale'].shape is not correct, expected: %s, actual: %s" % ((weight.value.shape[0],), result_q_param.ext["scale"].shape)
        assert result_q_param.ext["offset"].shape == (weight.value.shape[0],), \
            "q_param.ext['offset'].shape is not correct, expected: %s, actual: %s" % ((weight.value.shape[0],), result_q_param.ext["offset"].shape)
        assert result_q_param.ext["offset"].max() == 0 and result_q_param.ext["offset"].min() == 0, \
            "q_param.ext['offset'] is not correct, expected: %s, actual: %s" % (True, result_q_param.ext["offset"].max() == 0 and result_q_param.ext["offset"].min() == 0)

    def test_asymmetric_quantization(self):
        """测试非对称量化"""
        from msmodelslim.core.quantizer.impl.ssz import ssz_calculate_qparam

        # 创建测试权重
        weight = QStorage(QDType.FLOAT, torch.randn(10, 20))

        # 创建初始量化参数
        initial_q_param = calculate_qparam(
            min_val=torch.min(weight.T.value, dim=0)[0],
            max_val=torch.max(weight.T.value, dim=0)[0],
            q_dtype=QDType(self.asymmetric_config.dtype),
            q_scope=QScope(self.asymmetric_config.scope),
            symmetric=self.asymmetric_config.symmetric,
        )

        # 调用ssz_calculate_qparam
        result_q_param = ssz_calculate_qparam(weight.T, initial_q_param)

        # 验证返回的q_param
        assert result_q_param is not None, \
            "q_param is not correct, expected: %s, actual: %s" % (None, result_q_param)
        assert result_q_param.scheme == self.asymmetric_config.to_scheme(), \
            "q_param.scheme is not correct, expected: %s, actual: %s" % (self.asymmetric_config.to_scheme(), result_q_param.scheme)
        assert "scale" in result_q_param.ext, \
            "q_param.ext is not correct, expected: %s, actual: %s" % ("scale", result_q_param.ext.keys())
        assert "offset" in result_q_param.ext, \
            "q_param.ext is not correct, expected: %s, actual: %s" % ("offset", result_q_param.ext.keys())
        assert isinstance(result_q_param.ext["scale"], torch.Tensor), \
            "q_param.ext['scale'] is not correct, expected: %s, actual: %s" % (torch.Tensor, type(result_q_param.ext["scale"]))
        assert isinstance(result_q_param.ext["offset"], torch.Tensor), \
            "q_param.ext['offset'] is not correct, expected: %s, actual: %s" % (torch.Tensor, type(result_q_param.ext["offset"]))
        assert result_q_param.ext["scale"].shape == (weight.value.shape[0],), \
            "q_param.ext['scale'].shape is not correct, expected: %s, actual: %s" % ((weight.value.shape[0],), result_q_param.ext["scale"].shape)
        assert result_q_param.ext["offset"].shape == (weight.value.shape[0],), \
            "q_param.ext['offset'].shape is not correct, expected: %s, actual: %s" % ((weight.value.shape[0],), result_q_param.ext["offset"].shape)
        assert result_q_param.ext["offset"].max() != 0 or result_q_param.ext["offset"].min() != 0, \
            "q_param.ext['offset'] is not correct, expected: %s, actual: %s" % (True, result_q_param.ext["offset"].max() != 0 or result_q_param.ext["offset"].min() != 0)

    def test_scale_offset_validity(self):
        """测试scale和offset的有效性"""
        from msmodelslim.core.quantizer.impl.ssz import ssz_calculate_qparam

        weight = QStorage(QDType.FLOAT, torch.randn(10, 20))

        initial_q_param = calculate_qparam(
            min_val=torch.min(weight.T.value, dim=0)[0],
            max_val=torch.max(weight.T.value, dim=0)[0],
            q_dtype=QDType(self.symmetric_config.dtype),
            q_scope=QScope(self.symmetric_config.scope),
            symmetric=self.symmetric_config.symmetric,
        )

        result_q_param = ssz_calculate_qparam(weight.T, initial_q_param)

        # 验证scale不为零且为有限值
        assert torch.all(torch.isfinite(result_q_param.ext["scale"])), \
            "scale is not correct, expected: %s, actual: %s" % (True, torch.all(torch.isfinite(result_q_param.ext["scale"])))
        assert torch.all(result_q_param.ext["scale"] != 0), \
            "scale is not correct, expected: %s, actual: %s" % (True, torch.all(result_q_param.ext["scale"] != 0))

        # 验证offset为有限值
        assert torch.all(torch.isfinite(result_q_param.ext["offset"])), \
            "offset is not correct, expected: %s, actual: %s" % (True, torch.all(torch.isfinite(result_q_param.ext["offset"])))
