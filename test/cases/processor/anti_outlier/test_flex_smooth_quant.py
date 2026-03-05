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

# 标准库导入
from unittest.mock import Mock

# 第三方库导入
import torch

# 应用程序自定义模块导入
from msmodelslim.processor.anti_outlier.common.subgraph_type import (
    UpDownSubgraph,
)
from msmodelslim.processor.anti_outlier.common.smooth_types import FlexSmoothQuantConfig
from msmodelslim.processor.anti_outlier.common import SmoothContext
from msmodelslim.processor.anti_outlier.common.subgraph_type import LinearLinearSubgraph, NormLinearSubgraph, OVSubgraph
from msmodelslim.processor.anti_outlier.flex_smooth.alpha_beta_search import (
    FlexSmoothAlphaBetaSearcher,
    quant_int8asym,
    quant_int8sym,
)
from msmodelslim.processor.anti_outlier.common.scale_computation import (
    FlexSmoothScaleCalculator,
    MQGAScaleParams,
    apply_smooth_scale_shift,
    prepare_mqga_parameters,
    reduce_scales_for_mqga_max,
    reduce_scales_for_mqga_mean,
)
from msmodelslim.processor.anti_outlier.flex_smooth.api import (
    flex_smooth_impl_linear_linear,
    flex_smooth_impl_norm_linear,
    flex_smooth_impl_ov,
    flex_smooth_impl_up_down,
)


class TestQuantizationFunctions:
    """测试量化相关的基础函数"""

    @staticmethod
    def test_quant_int8sym_basic():
        """测试对称int8量化的基本功能"""
        # 测试正常情况
        x = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        result = quant_int8sym(x)
        
        # 验证输出形状
        assert result.shape == x.shape
        # 验证数据类型
        assert result.dtype == x.dtype
        # 验证值在合理范围内
        assert torch.all(result >= -127)
        assert torch.all(result <= 127)

    @staticmethod
    def test_quant_int8tasym_basic():
        """测试非对称int8量化的基本功能"""
        # 测试正常情况
        x = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        result = quant_int8asym(x)
        
        # 验证输出形状
        assert result.shape == x.shape
        # 验证数据类型
        assert result.dtype == x.dtype

    @staticmethod
    def test_quant_int8tasym_edge_cases():
        """测试非对称int8量化的边界情况"""
        # 测试零张量
        x_zero = torch.zeros(2, 3)
        result_zero = quant_int8asym(x_zero)
        assert torch.allclose(result_zero, x_zero)
        
        # 测试负值
        x_neg = torch.tensor([[-1.0, -2.0], [-3.0, -4.0]])
        result_neg = quant_int8asym(x_neg)
        assert result_neg.shape == x_neg.shape
        
        # 测试单值张量
        x_single = torch.tensor([5.0])
        result_single = quant_int8asym(x_single)
        assert result_single.shape == x_single.shape

    @staticmethod
    def test_scale_descale_basic():
        """测试尺度缩放和反缩放的基本功能"""
        act = torch.randn(10, 8)
        fc_weights = torch.randn(4, 8)
        alpha = 0.5
        beta = 0.5
        
        searcher = FlexSmoothAlphaBetaSearcher(act_sym=True, search_step=0.05)
        result = searcher.evaluate_alpha_beta(act, fc_weights, alpha, beta)
        
        # 验证返回的是标量
        assert isinstance(result, float)
        assert result >= 0  # MSE应该非负

    @staticmethod
    def test_scale_descale_with_asym():
        """测试非对称激活的尺度缩放"""
        act = torch.randn(10, 8)
        fc_weights = torch.randn(4, 8)
        alpha = 0.3
        beta = 0.7
        
        # 测试对称激活
        searcher_sym = FlexSmoothAlphaBetaSearcher(act_sym=True, search_step=0.05)
        result_sym = searcher_sym.evaluate_alpha_beta(act, fc_weights, alpha, beta)
        # 测试非对称激活
        searcher_asym = FlexSmoothAlphaBetaSearcher(act_sym=False, search_step=0.05)
        result_asym = searcher_asym.evaluate_alpha_beta(act, fc_weights, alpha, beta)
        
        assert isinstance(result_sym, float)
        assert isinstance(result_asym, float)
        assert result_sym >= 0
        assert result_asym >= 0

    @staticmethod
    def test_search_alpha_beta_basic():
        """测试alpha和beta搜索的基本功能"""
        act = torch.randn(10, 8)
        fc_weights = torch.randn(4, 8)
        
        searcher = FlexSmoothAlphaBetaSearcher(act_sym=True, search_step=0.05)
        best_alpha, best_beta, best_mse = searcher.search_alpha_beta(act, fc_weights)
        
        # 验证返回值
        assert isinstance(best_alpha, float)
        assert isinstance(best_beta, float)
        assert isinstance(best_mse, float)
        assert 0.0 <= best_alpha <= 1.0
        assert 0.0 <= best_beta <= 1.0
        assert best_mse >= 0

    @staticmethod
    def test_search_alpha_beta_with_best_alpha():
        """测试给定最佳alpha时的beta搜索"""
        act = torch.randn(10, 8)
        fc_weights = torch.randn(4, 8)
        best_alpha = 0.5
        
        searcher = FlexSmoothAlphaBetaSearcher(act_sym=True, search_step=0.05)
        # 注意：新接口不支持直接指定best_alpha，这里测试固定alpha的情况
        best_beta, best_mse = searcher.search_beta(act, fc_weights, best_alpha)
        
        assert isinstance(best_beta, float)
        assert isinstance(best_mse, float)
        assert 0.0 <= best_beta <= 1.0
        assert best_mse >= 0

    @staticmethod
    def test_compute_smooth_scale_basic():
        """测试平滑尺度计算的基本功能"""
        a_scale = torch.tensor([1.0, 2.0, 3.0])
        w_scale = torch.tensor([0.5, 1.5, 2.5])
        alpha = 0.5
        beta = 0.5
        
        calculator = FlexSmoothScaleCalculator(alpha=alpha, beta=beta)
        result = calculator.compute_smooth_scale(a_scale, w_scale)
        
        assert result.shape == a_scale.shape
        assert torch.all(result > 0)  # 尺度应该为正
        assert result.dtype == a_scale.dtype

    @staticmethod
    def test_compute_smooth_scale_edge_cases():
        """测试平滑尺度计算的边界情况"""
        # 测试零值
        a_scale = torch.tensor([0.0, 1.0, 2.0])
        w_scale = torch.tensor([1.0, 0.0, 1.0])
        alpha = 0.5
        beta = 0.5
        
        calculator = FlexSmoothScaleCalculator(alpha=alpha, beta=beta)
        result = calculator.compute_smooth_scale(a_scale, w_scale)
        assert torch.all(result >= 1e-5)  # 应该应用最小阈值

    @staticmethod
    def test_apply_smooth_scale_shift():
        """测试平滑尺度应用"""
        # 创建模拟层
        layer = Mock()
        layer.weight = torch.randn(8, 4)
        original_weight = layer.weight.clone()
        
        scales = torch.tensor([1.0, 2.0, 3.0, 4.0])
        
        apply_smooth_scale_shift(layer, scales)
        
        # 验证权重被修改
        assert not torch.allclose(layer.weight, original_weight)

    @staticmethod
    def test_prepare_mqga_parameters():
        """测试MQGA参数准备"""
        num_attention_heads = 8
        num_key_value_heads = 2
        
        ratio, pad_size = prepare_mqga_parameters(num_attention_heads, num_key_value_heads)
        
        assert ratio == 4  # 8 // 2
        assert pad_size == 0

    @staticmethod
    def test_reduce_scales_for_mqga_max():
        """测试MQGA尺度缩减（使用max聚合）
        
        测试场景：
        - 8个Q头，2个KV头（shape_ratio=4）
        - 每个头的维度为128
        - 总维度：8 * 128 = 1024
        """
        # 设置参数
        num_attention_heads = 8
        num_kv_heads = 2
        head_dim = 128
        num_key_value_groups = num_attention_heads // num_kv_heads  # 4
        total_dim = num_attention_heads * head_dim  # 1024
        
        # 创建测试数据
        act_scales = torch.randn(total_dim).abs() + 0.1  # 确保为正值
        weight_scales = torch.randn(total_dim).abs() + 0.1  # 确保为正值
        best_alpha = 0.5
        best_beta = 0.5
        
        # 封装参数为 MQGAScaleParams 对象
        params = MQGAScaleParams(
            act_scales=act_scales,
            weight_scales=weight_scales,
            best_alpha=best_alpha,
            best_beta=best_beta,
            num_key_value_groups=num_key_value_groups,
            head_dim=head_dim
        )
        
        # 调用函数（使用max聚合）
        o_scales, v_scales = reduce_scales_for_mqga_max(params)
        
        # 验证输出维度
        assert o_scales.shape == (total_dim,), f"o_scales维度应为{total_dim}，实际为{o_scales.shape}"
        assert v_scales.shape == (num_kv_heads * head_dim,), \
            f"v_scales维度应为{num_kv_heads * head_dim}，实际为{v_scales.shape}"
        
        # 验证o_scales是v_scales的重复扩展
        assert o_scales.numel() == num_key_value_groups * v_scales.numel(), \
            "o_scales应该是v_scales重复num_key_value_groups次"
        
        # 验证数值范围合理（应该都是正值且被clamp到最小1e-5）
        assert torch.all(o_scales > 0), "o_scales应该都是正值"
        assert torch.all(v_scales > 0), "v_scales应该都是正值"
        assert torch.all(o_scales >= 1e-5), "o_scales应该被clamp到最小1e-5"
        assert torch.all(v_scales >= 1e-5), "v_scales应该被clamp到最小1e-5"

    @staticmethod
    def test_reduce_scales_for_mqga_mean():
        """测试MQGA尺度缩减（使用mean聚合）
        
        测试场景：
        - 8个Q头，2个KV头（shape_ratio=4）
        - 每个头的维度为128
        - 总维度：8 * 128 = 1024
        """
        # 设置参数
        num_attention_heads = 8
        num_kv_heads = 2
        head_dim = 128
        shape_ratio = num_attention_heads // num_kv_heads  # 4
        total_dim = num_attention_heads * head_dim  # 1024
        
        # 创建测试数据（已经计算好的scales）
        scales = torch.randn(total_dim).abs() + 0.1  # 确保为正值
        
        # 调用函数（使用mean聚合）
        o_scales, v_scales = reduce_scales_for_mqga_mean(scales, shape_ratio, num_attention_heads)
        
        # 验证输出维度
        assert o_scales.shape == scales.shape, f"o_scales维度应为{scales.shape}，实际为{o_scales.shape}"
        assert v_scales.numel() == scales.numel() // shape_ratio, \
            f"v_scales元素数应为{scales.numel() // shape_ratio}，实际为{v_scales.numel()}"
        
        # 验证o_scales是v_scales的重复扩展
        assert o_scales.numel() == shape_ratio * v_scales.numel(), \
            "o_scales应该是v_scales重复shape_ratio次"
        
        # 验证数值范围合理（应该都是正值）
        assert torch.all(o_scales > 0), "o_scales应该都是正值"
        assert torch.all(v_scales > 0), "v_scales应该都是正值"


class TestFlexSmoothImplOV:
    """测试OV子图的平滑实现"""

    @staticmethod
    def create_mock_ov_subgraph():
        """创建模拟的OV子图"""
        subgraph = Mock(spec=OVSubgraph)
        subgraph.v_proj = Mock()
        subgraph.o_proj = Mock()
        subgraph.num_attention_heads = 8
        subgraph.key_value_heads = 2
        
        # 设置权重
        subgraph.o_proj.weight = torch.randn(8, 16)
        subgraph.v_proj.weight = torch.randn(4, 8)
        
        # 设置参数 - 返回迭代器而不是列表
        subgraph.v_proj.parameters.return_value = iter([torch.randn(16, 8)])
        
        return subgraph

    @staticmethod
    def create_mock_context():
        """创建模拟的平滑上下文"""
        context = Mock(spec=SmoothContext)
        context.tensors = [torch.randn(2, 8, 16)]
        context.a_smooth_scale = torch.randn(16)
        return context

    @staticmethod
    def create_mock_config(alpha=None, beta=None):
        """创建模拟的配置"""
        config = Mock(spec=FlexSmoothQuantConfig)
        config.alpha = alpha
        config.beta = beta
        return config

    @staticmethod
    def test_flex_smooth_impl_ov_basic():
        """测试OV平滑实现的基本功能"""
        subgraph = TestFlexSmoothImplOV.create_mock_ov_subgraph()
        context = TestFlexSmoothImplNormLinear.create_mock_context()
        config = TestFlexSmoothImplNormLinear.create_mock_config()
        
        # 应该正常执行而不抛出异常
        flex_smooth_impl_ov(subgraph, config, context)

    @staticmethod
    def test_flex_smooth_impl_ov_with_provided_params():
        """测试使用提供的alpha和beta参数"""
        subgraph = TestFlexSmoothImplOV.create_mock_ov_subgraph()
        context = TestFlexSmoothImplOV.create_mock_context()
        config = TestFlexSmoothImplOV.create_mock_config(alpha=0.5, beta=0.5)
        
        flex_smooth_impl_ov(subgraph, config, context)


class TestFlexSmoothImplUpDown:
    """测试Up-Down子图的平滑实现"""

    @staticmethod
    def create_mock_updown_subgraph():
        """创建模拟的Up-Down子图"""
        subgraph = Mock(spec=UpDownSubgraph)
        subgraph.up_proj = Mock()
        subgraph.down_proj = Mock()
        subgraph.gate_proj = None  # 可选
        
        # 设置权重
        subgraph.down_proj.weight = torch.randn(8, 16)
        subgraph.up_proj.weight = torch.randn(16, 8)
        
        # 设置参数
        subgraph.up_proj.parameters.return_value = iter([torch.randn(16, 8)])
        
        return subgraph

    @staticmethod
    def create_mock_context():
        """创建模拟的平滑上下文"""
        context = Mock(spec=SmoothContext)
        context.tensors = [torch.randn(2, 8, 16)]
        context.a_smooth_scale = torch.randn(16)
        return context

    @staticmethod
    def create_mock_config(alpha=None, beta=None):
        """创建模拟的配置"""
        config = Mock(spec=FlexSmoothQuantConfig)
        config.alpha = alpha
        config.beta = beta
        return config

    @staticmethod
    def test_flex_smooth_impl_updown_basic():
        """测试Up-Down平滑实现的基本功能"""
        subgraph = TestFlexSmoothImplUpDown.create_mock_updown_subgraph()
        context = TestFlexSmoothImplNormLinear.create_mock_context()
        config = TestFlexSmoothImplNormLinear.create_mock_config()
        
        flex_smooth_impl_up_down(subgraph, config, context)

    @staticmethod
    def test_flex_smooth_impl_updown_with_gate_proj():
        """测试包含gate_proj的Up-Down平滑实现"""
        subgraph = TestFlexSmoothImplUpDown.create_mock_updown_subgraph()
        subgraph.gate_proj = Mock()
        context = TestFlexSmoothImplNormLinear.create_mock_context()
        config = TestFlexSmoothImplNormLinear.create_mock_config()
        
        flex_smooth_impl_up_down(subgraph, config, context)


class TestFlexSmoothImplLinearLinear:
    """测试Linear-Linear子图的平滑实现"""

    @staticmethod
    def create_mock_linearlinear_subgraph():
        """创建模拟的Linear-Linear子图"""
        subgraph = Mock(spec=LinearLinearSubgraph)
        subgraph.linear1 = Mock()
        subgraph.linear2 = Mock()
        
        # 设置权重
        subgraph.linear2.weight = torch.randn(8, 16)
        subgraph.linear1.weight = torch.randn(16, 8)
        
        # 设置参数
        subgraph.linear1.parameters.return_value = iter([torch.randn(16, 8)])
        
        return subgraph

    @staticmethod
    def create_mock_context():
        """创建模拟的平滑上下文"""
        context = Mock(spec=SmoothContext)
        context.tensors = [torch.randn(2, 8, 16)]
        context.a_smooth_scale = torch.randn(16)
        return context

    @staticmethod
    def create_mock_config(alpha=None, beta=None):
        """创建模拟的配置"""
        config = Mock(spec=FlexSmoothQuantConfig)
        config.alpha = alpha
        config.beta = beta
        return config

    @staticmethod
    def test_flex_smooth_impl_linearlinear_basic():
        """测试Linear-Linear平滑实现的基本功能"""
        subgraph = TestFlexSmoothImplLinearLinear.create_mock_linearlinear_subgraph()
        context = TestFlexSmoothImplNormLinear.create_mock_context()
        config = TestFlexSmoothImplNormLinear.create_mock_config()
        
        flex_smooth_impl_linear_linear(subgraph, config, context)


class TestFlexSmoothImplNormLinear:
    """测试Norm-Linear子图的平滑实现"""

    @staticmethod
    def create_mock_normlinear_subgraph():
        """创建模拟的Norm-Linear子图"""
        subgraph = Mock(spec=NormLinearSubgraph)
        subgraph.norm = Mock()
        subgraph.linears = [Mock(), Mock()]  # 多个线性层
        subgraph.linear_names = ["name0", "name1"]
        
        # 设置权重
        for linear in subgraph.linears:
            linear.weight = torch.randn(8, 16)
        
        # 设置参数
        subgraph.norm.parameters.return_value = iter([torch.randn(16)])
        
        return subgraph

    @staticmethod
    def create_mock_context():
        """创建模拟的平滑上下文"""
        context = Mock(spec=SmoothContext)
        context.tensors = [torch.randn(2, 8, 16)]
        context.a_smooth_scale = torch.randn(16)
        return context

    @staticmethod
    def create_mock_config(alpha=None, beta=None):
        """创建模拟的配置"""
        config = Mock(spec=FlexSmoothQuantConfig)
        config.alpha = alpha
        config.beta = beta
        return config

    @staticmethod
    def test_flex_smooth_impl_normlinear_basic():
        """测试Norm-Linear平滑实现的基本功能"""
        subgraph = TestFlexSmoothImplNormLinear.create_mock_normlinear_subgraph()
        context = TestFlexSmoothImplNormLinear.create_mock_context()
        config = TestFlexSmoothImplNormLinear.create_mock_config()
        
        flex_smooth_impl_norm_linear(subgraph, config, context)
