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

from msmodelslim.ir.qal.qbase import QDType, QScheme, QScope
from msmodelslim import ir as qir
from msmodelslim.core.observer.histogram import HistogramObserver, HistogramObserverConfig, SearchMethod
from msmodelslim.core.quantizer.base import QConfig, AutoActQuantizer
from msmodelslim.core.quantizer.impl.histogram import ActPerTensorHistogram
from msmodelslim.utils.exception import SpecError, UnexpectedError, SchemaValidateError


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


class TestHistogramObserverConfig:
    """测试直方图观察器配置类"""
    
    def test_default_config(self):
        """测试默认配置"""
        config = HistogramObserverConfig()
        
        assert config.symmetric is False
        assert config.search_method == SearchMethod.L2_NORM
        assert config.dtype == QDType.INT8
        assert config.scope == QScope.PER_TENSOR
    
    def test_custom_config(self):
        """测试自定义配置"""
        config = HistogramObserverConfig(
            symmetric=True,
            search_method=SearchMethod.KL_DIVERGENCE,
            dtype=QDType.INT8,
            scope=QScope.PER_TENSOR)
        assert config.symmetric is True
        assert config.search_method == SearchMethod.KL_DIVERGENCE
        assert config.dtype == "int8"
        assert config.scope == "per_tensor"

    def test_invalid_config(self):
        """测试无效配置"""
        with pytest.raises(SchemaValidateError):
            HistogramObserverConfig(
                symmetric=True,
                search_method=SearchMethod.KL_DIVERGENCE,
                dtype="int16", # 不支持int16
                scope=QScope.PER_CHANNEL
            )
        

class TestHistogramObserver:
    """测试直方图观察器"""
    
    def test_initialization(self):
        """测试初始化"""
        config = HistogramObserverConfig()
        observer = HistogramObserver(config)
        
        assert observer.config == config
        assert observer.clip_min is None
        assert observer.clip_max is None
        assert observer.upsample_rate == 16
    
    def test_update_with_valid_input(self):
        """测试有效输入更新"""
        config = HistogramObserverConfig()
        observer = HistogramObserver(config)
        
        x = torch.randn(10, 10)
        observer.update(x)
        
        # 验证clip bounds被设置
        clip_min, clip_max = observer.get_clip_bounds()
        assert isinstance(clip_min, torch.Tensor)
        assert isinstance(clip_max, torch.Tensor)
        assert not torch.isinf(clip_min)
        assert not torch.isinf(clip_max)
        assert not torch.isnan(clip_min)
        assert not torch.isnan(clip_max)
    
    def test_update_with_same_values(self):
        """测试所有值相同的情况"""
        config = HistogramObserverConfig()
        observer = HistogramObserver(config)
        
        x = torch.ones(10, 10) * 5.0
        observer.update(x)
        
        # 验证clip bounds被正确设置
        clip_min, clip_max = observer.get_clip_bounds()
        assert isinstance(clip_min, torch.Tensor)
        assert isinstance(clip_max, torch.Tensor)
        assert not torch.isinf(clip_min)
        assert not torch.isinf(clip_max)
        assert not torch.isnan(clip_min)
        assert not torch.isnan(clip_max)
        
    
    def test_update_with_edge_cases(self):
        """测试边界情况"""
        config = HistogramObserverConfig()
        observer = HistogramObserver(config)
        
        # 测试空张量
        x = torch.tensor([])
        with pytest.raises(SpecError, match="Input tensor is empty"):
            observer.update(x)
        
        # 测试None输入
        with pytest.raises(SpecError, match="Input must be a valid torch.Tensor"):
            observer.update(None)
        
        # 测试非张量输入
        with pytest.raises(SpecError, match="Input must be a valid torch.Tensor"):
            observer.update([1, 2, 3])
        
        # 测试只包含inf的输入
        x = torch.tensor([float('inf'), float('-inf')])
        with pytest.raises(SpecError, match="Input tensor is empty"):
            observer.update(x)
        
        # 测试只包含nan的输入
        x = torch.tensor([float('nan'), float('nan')])
        with pytest.raises(SpecError, match="Input tensor is empty"):
            observer.update(x)

    def test_get_clip_bounds_before_update(self):
        """测试在update之前获取clip bounds"""
        config = HistogramObserverConfig()
        observer = HistogramObserver(config)
        
        with pytest.raises(SpecError, match="Clip min or clip max is not set"):
            observer.get_clip_bounds()

    def test_reset(self):
        """测试重置功能"""
        config = HistogramObserverConfig()
        observer = HistogramObserver(config)
        
        # 更新观察器
        x = torch.randn(10, 10)
        observer.update(x)
        
        # 验证状态已更新
        clip_min, clip_max = observer.get_clip_bounds()
        assert isinstance(clip_min, torch.Tensor)
        assert isinstance(clip_max, torch.Tensor)
        assert not torch.isinf(clip_min)
        assert not torch.isnan(clip_min)
        assert not torch.isinf(clip_max)
        assert not torch.isnan(clip_max)
        
        # 重置观察器
        observer.reset()
        
        # 验证重置后可以再次更新
        x = torch.randn(10, 10)
        observer.update(x)
        clip_min, clip_max = observer.get_clip_bounds()
        assert isinstance(clip_min, torch.Tensor)
        assert isinstance(clip_max, torch.Tensor)
        assert not torch.isinf(clip_min)
        assert not torch.isinf(clip_max)
        assert not torch.isnan(clip_min)
        assert not torch.isnan(clip_max)

        observer.reset()
        
        # 验证状态已重置
        with pytest.raises(SpecError, match="Clip min or clip max is not set"):
            observer.get_clip_bounds()
    
    def test_compute_quantization_error_l2(self):
        """测试L2误差计算"""
        config = HistogramObserverConfig(search_method=SearchMethod.L2_NORM)
        observer = HistogramObserver(config)
        
        # 设置必要的状态
        x = torch.randn(10, 10)
        observer.update(x)
        
        # 测试L2误差计算
        error = observer._compute_quantization_error(0, 100)
        assert isinstance(error, float)
        assert error >= 0.0
    
    def test_compute_quantization_error_kl(self):
        """测试KL散度误差计算"""
        config = HistogramObserverConfig(search_method=SearchMethod.KL_DIVERGENCE)
        observer = HistogramObserver(config)
        
        # 设置必要的状态
        x = torch.randn(10, 10)
        observer.update(x)
        error = observer._compute_quantization_error(0, 100)
        assert isinstance(error, float)
        assert error >= 0.0
    
    def test_non_linear_param_search(self):
        """测试非线性参数搜索"""
        config = HistogramObserverConfig(search_method=SearchMethod.L2_NORM)
        observer = HistogramObserver(config)
        
        # 设置必要的状态
        x = torch.randn(10, 10)
        observer.update(x)
        
        # 测试参数搜索
        new_min, new_max = observer._non_linear_param_search()
        assert isinstance(new_min, torch.Tensor)
        assert isinstance(new_max, torch.Tensor)
        assert new_min <= new_max
    
    def test_non_linear_param_search_empty_histogram(self):
        """测试因为溢出产生的空直方图的参数搜索"""
        config = HistogramObserverConfig()
        observer = HistogramObserver(config)
        
        # 设置空直方图
        x = torch.tensor([torch.finfo(torch.float32).max,torch.finfo(torch.float32).min]).to(torch.float32)
        observer.update(x)
        
        # 应该返回原始值
        new_min, new_max = observer._non_linear_param_search()
        assert new_min == torch.tensor(torch.finfo(torch.float32).min)
        assert new_max == torch.tensor(torch.finfo(torch.float32).max)
    
    def test_non_linear_param_search_bins_mismatch(self):
        """测试bins不匹配的情况"""
        config = HistogramObserverConfig()
        observer = HistogramObserver(config)
        
        # 设置不匹配的bins
        observer.histogram = torch.ones(1024)  # 不同的bins数量
        observer.min_val = torch.tensor(0.0)
        observer.max_val = torch.tensor(1.0)
        observer.bins = 2048
        
        with pytest.raises(UnexpectedError, match="Histogram bins mismatch"):
            observer._non_linear_param_search()

class TestActPerTensorHistogram:
    """测试Per-Tensor激活Histogram量化器"""

    def test_initialization(self):
        """测试初始化"""
        config = QConfig(
            dtype="int8",
            scope="per_tensor",
            method="histogram",
            symmetric=True
        )

        quantizer = ActPerTensorHistogram(config)

        assert quantizer.config == config
        assert isinstance(quantizer.histogram_observer, HistogramObserver)
        assert quantizer.q_param is None

    def test_forward_then_can_get_correct_q_param(self):
        """测试前向传播并验证量化参数"""
        config = QConfig(
            dtype="int8",
            scope="per_tensor",
            method="histogram",
            symmetric=True
        )

        quantizer = ActPerTensorHistogram(config)

        # 测试输入
        x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])

        result = quantizer(x)

        # 验证q_param被设置
        q_param = quantizer.get_q_param()
        assert q_param
        assert q_param.scheme == config.to_scheme()
        assert isinstance(q_param.ext, dict)
        assert "scale" in q_param.ext
        assert "offset" in q_param.ext
        assert isinstance(q_param.ext["scale"], torch.Tensor)
        assert isinstance(q_param.ext["offset"], torch.Tensor)
        assert q_param.ext["scale"].shape == (1,)
        assert q_param.ext["offset"].shape == (1,)

        # 验证输出形状
        assert result.shape == x.shape

    def test_forward_with_batch_input(self):
        """测试批量输入"""
        config = QConfig(
            dtype="int8",
            scope="per_tensor",
            method="histogram",
            symmetric=True
        )

        quantizer = ActPerTensorHistogram(config)

        # 测试不同形状的输入
        x = torch.randn(2, 3, 4)  # (batch, seq, hidden)
        result = quantizer(x)
        assert result.shape == x.shape

    @pytest.mark.parametrize(
        "qconfig",
        [
            to_qconfig(qir.int8_per_tensor_sym, "histogram"),
            to_qconfig(qir.int8_per_tensor_asym, "histogram"),
        ]
    )
    def test_creation_with_auto_quantizer(self, qconfig):
        """测试通过自动量化器创建"""
        quantizer = AutoActQuantizer.from_config(qconfig)
        assert isinstance(quantizer, ActPerTensorHistogram)

    def test_get_q_param_before_forward(self):
        """测试在forward之前获取q_param应该失败"""
        config = QConfig(
            dtype="int8",
            scope="per_tensor",
            method="histogram",
            symmetric=True
        )

        quantizer = ActPerTensorHistogram(config)

        with pytest.raises(SpecError, match="No q_param was set"):
            quantizer.get_q_param()
    def test_get_q_param_after_forward(self):
        """测试在forward之后可以正常获取q_param"""
        config = QConfig(
            dtype="int8",
            scope="per_tensor",
            method="histogram",
            symmetric=True
        )
        quantizer = ActPerTensorHistogram(config)
        x = torch.randn(8, 8)
        quantizer(x)
        q_param = quantizer.get_q_param()
        assert q_param is not None
        assert isinstance(q_param.ext, dict)
        assert "scale" in q_param.ext
        assert "offset" in q_param.ext
        assert isinstance(q_param.ext["scale"], torch.Tensor)
        assert isinstance(q_param.ext["offset"], torch.Tensor)
        assert q_param.scheme == config.to_scheme()

    def test_forward_with_edge_cases(self):
        """测试边界情况"""
        config = QConfig(
            dtype="int8",
            scope="per_tensor",
            method="histogram",
            symmetric=True
        )

        quantizer = ActPerTensorHistogram(config)

        # 测试空张量
        x = torch.tensor([])
        with pytest.raises(SpecError, match="Input tensor is empty"):
            quantizer(x)

        # 测试None输入
        with pytest.raises(SpecError, match="Input must be a valid torch.Tensor"):
            quantizer(None)

        # 测试非张量输入
        with pytest.raises(SpecError, match="Input must be a valid torch.Tensor"):
            quantizer([1, 2, 3])

        # 测试只包含inf的输入
        x = torch.tensor([float('inf'), float('-inf')])
        with pytest.raises(SpecError, match="Input tensor is empty"):
            quantizer(x)

        # 测试只包含nan的输入
        x = torch.tensor([float('nan'), float('nan')])
        with pytest.raises(SpecError, match="Input tensor is empty"):
            quantizer(x)

    def test_forward_with_extreme_values(self):
        """测试极值输入"""
        config = QConfig(
            dtype="int8",
            scope="per_tensor",
            method="histogram",
            symmetric=True
        )

        quantizer = ActPerTensorHistogram(config)
        finfo_float32 = torch.finfo(torch.float32)
        # 测试极值输入
        x = torch.tensor([finfo_float32.min, finfo_float32.max, -finfo_float32.max]).to(torch.float32)
        result = quantizer(x)
        assert result.shape == x.shape
        assert not torch.isinf(result).any()
        assert not torch.isnan(result).any()

    def test_forward_with_same_values(self):
        """测试全等输入"""
        config = QConfig(
            dtype="int8",
            scope="per_tensor",
            method="histogram",
            symmetric=True
        )

        quantizer = ActPerTensorHistogram(config)

        # 测试全等输入
        x = torch.ones(10, 10)
        result = quantizer(x)
        assert result.shape == x.shape

    def test_histogram_observer_integration(self):  
        """测试直方图观察器集成"""
        config = QConfig(
            dtype="int8",
            scope="per_tensor",
            method="histogram",
            symmetric=False
        )

        quantizer = ActPerTensorHistogram(config)
        
        # 验证观察器配置
        assert quantizer.histogram_observer.config.search_method == SearchMethod.L2_NORM
        assert quantizer.histogram_observer.config.dtype == QDType.INT8
        assert quantizer.histogram_observer.config.scope == QScope.PER_TENSOR
        assert quantizer.histogram_observer.config.symmetric == False
        
        # 测试观察器更新
        x = torch.randn(10, 10)
        quantizer(x)
        
        # 验证观察器状态
        clip_min, clip_max = quantizer.histogram_observer.get_clip_bounds()
        assert isinstance(clip_min, torch.Tensor)
        assert isinstance(clip_max, torch.Tensor)
        assert not torch.isnan(clip_min)
        assert not torch.isnan(clip_max)
        assert not torch.isinf(clip_min)
        assert not torch.isinf(clip_max)

    def test_quantizer_symmetric_vs_asymmetric(self):
        """测试对称和非对称量化"""
        for symmetric in [True, False]:
            config = QConfig(
                dtype="int8",
                scope="per_tensor",
                method="histogram",
                symmetric=symmetric
            )
            quantizer = ActPerTensorHistogram(config)
            
            x = torch.randn(5, 5)
            result = quantizer(x)
            assert result.shape == x.shape
            
            q_param = quantizer.get_q_param()
            assert q_param.scheme.symmetric == symmetric