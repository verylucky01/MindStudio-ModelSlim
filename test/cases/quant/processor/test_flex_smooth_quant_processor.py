# -*- coding: utf-8 -*-

# Copyright (c) 2025-2025 Huawei Technologies Co., Ltd.
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

from functools import partial
from typing import Dict, List
from unittest.mock import Mock, patch

import pytest
import torch
import torch.nn as nn
from msmodelslim.core.base.protocol import BatchProcessRequest
from msmodelslim.quant.processor.anti_outlier.flex_smooth import (
    FlexSmoothQuantProcessor,
    FlexSmoothQuantProcessorConfig
)
from msmodelslim.quant.processor.anti_outlier.common.smooth_components import StatKey
from msmodelslim.quant.processor.anti_outlier.flex_smooth.interface import FlexSmoothQuantInterface
from msmodelslim.utils.exception import UnsupportedError
from msmodelslim.core.graph.adapter_types import AdapterConfig, MappingConfig, FusionConfig



class MockModel(nn.Module):
    """Mock model for testing"""
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(10, 20)
        self.layer2 = nn.Linear(20, 30)
        self.norm_layer = nn.LayerNorm(10)
        
    def get_submodule(self, name):
        """Mock get_submodule method"""
        if name == "layer1":
            return self.layer1
        elif name == "layer2":
            return self.layer2
        elif name == "model.layers.0.input_layernorm":
            return self.norm_layer
        return None
    
    def set_submodule(self, name, module):
        """Mock set_submodule method"""
        if name == "model.layers.0.input_layernorm":
            self.norm_layer = module
    
    def named_modules(self):
        """Mock named_modules method"""
        modules = [
            ('layer1', self.layer1),
            ('layer2', self.layer2), 
            ('norm_layer', self.norm_layer)
        ]
        for name, module in modules:
            yield name, module


class MockFlexSmoothQuantInterface(FlexSmoothQuantInterface):
    """Mock adapter implementing FlexSmoothQuantInterface"""
    def get_adapter_config_for_subgraph(self) -> List[AdapterConfig]:
        return [
            AdapterConfig(
                subgraph_type="norm-linear",
                mapping=MappingConfig(source="model.layers.0.input_layernorm", targets=["layer1"])
            )
        ]


class TestFlexSmoothQuantProcessor:
    """FlexSmoothQuantProcessor test class"""

    def __init__(self):
        """Initialize test class attributes"""
        self.model = None
        self.adapter = None
        self.default_config = None

    @staticmethod
    def test_config_validation():
        """Test configuration validation"""
        # 测试 alpha 值验证
        with pytest.raises(Exception):
            invalid_config = FlexSmoothQuantProcessorConfig(alpha=-1.0)

        with pytest.raises(Exception):
            invalid_config = FlexSmoothQuantProcessorConfig(alpha=2.0)

        # 测试 beta 值验证
        with pytest.raises(Exception):
            invalid_config = FlexSmoothQuantProcessorConfig(beta=-0.5)

        with pytest.raises(Exception):
            invalid_config = FlexSmoothQuantProcessorConfig(beta=1.5)

    def setup_method(self):
        """Setup before tests"""
        self.model = MockModel()
        self.adapter = MockFlexSmoothQuantInterface()
        self.default_config = FlexSmoothQuantProcessorConfig(
            alpha=0.5,
            beta=0.8,
            enable_subgraph_type=["norm-linear", "linear-linear"]
        )

    def test_init_with_valid_adapter(self):
        """Test initialization with valid adapter"""
        processor = FlexSmoothQuantProcessor(
            model=self.model,
            config=self.default_config,
            adapter=self.adapter
        )

        assert processor.model == self.model
        assert processor.config == self.default_config
        assert processor.adapter == self.adapter
        assert isinstance(processor.stats_collector.act_stats, dict)
        assert processor.config.alpha == 0.5
        assert processor.config.beta == 0.8
        assert processor.config.enable_subgraph_type == ["norm-linear", "linear-linear"]

    def test_init_with_invalid_adapter(self):
        """Test initialization with invalid adapter"""
        invalid_adapter = Mock()

        with pytest.raises(UnsupportedError) as exc_info:
            FlexSmoothQuantProcessor(
                model=self.model,
                config=self.default_config,
                adapter=invalid_adapter
            )

        assert "does not implement FlexSmoothQuantInterface" in str(exc_info.value)

    def test_config_default_values(self):
        """Test configuration default values"""
        config = FlexSmoothQuantProcessorConfig()
        processor = FlexSmoothQuantProcessor(
            model=self.model,
            config=config,
            adapter=self.adapter
        )

        assert processor.config.alpha is None
        assert processor.config.beta is None
        assert processor.config.enable_subgraph_type == ["norm-linear", "linear-linear", "ov", "up-down"]

    def test_support_distributed(self):
        """Test distributed support"""
        processor = FlexSmoothQuantProcessor(
            model=self.model,
            config=self.default_config,
            adapter=self.adapter
        )

        assert processor.support_distributed() is True

    def test_preprocess(self):
        """Test preprocess method"""
        processor = FlexSmoothQuantProcessor(
            model=self.model,
            config=self.default_config,
            adapter=self.adapter
        )

        processor.global_adapter_config = self.adapter.get_adapter_config_for_subgraph()

        request = BatchProcessRequest(
            name="model.layers.0",
            module=self.model.layer1
        )

        # Preprocess method should execute normally without throwing exception
        processor.preprocess(request)

    def test_get_stats_hook(self):
        """Test statistics hook generation"""
        processor = FlexSmoothQuantProcessor(
            model=self.model,
            config=self.default_config,
            adapter=self.adapter
        )

        # Create hook function
        hook = processor.stats_collector.create_hook("test_module")
        assert callable(hook)

        # Test hook function execution
        mock_module = nn.Linear(10, 20)
        input_tensor = (torch.randn(5, 10),)
        output = torch.randn(5, 20)

        hook(mock_module, input_tensor, output)

        # Verify statistics data is updated
        assert "test_module" in processor.stats_collector.act_stats
        stats_dict = processor.stats_collector.act_stats["test_module"]
        assert StatKey.TENSOR in stats_dict
        assert StatKey.STAT_KEY_SMOOTH_SCALE in stats_dict

    def test_get_stats_hook_with_multiple_calls(self):
        """Test hook function with multiple calls"""
        processor = FlexSmoothQuantProcessor(
            model=self.model,
            config=self.default_config,
            adapter=self.adapter
        )

        hook = processor.stats_collector.create_hook("test_module")
        mock_module = nn.Linear(10, 20)

        # First call
        input_tensor1 = (torch.randn(3, 10),)
        hook(mock_module, input_tensor1, torch.randn(3, 20))

        # Second call
        input_tensor2 = (torch.randn(4, 10),)
        hook(mock_module, input_tensor2, torch.randn(4, 20))

        # Verify statistics after multiple calls
        stats_dict = processor.stats_collector.act_stats["test_module"]
        assert StatKey.TENSOR in stats_dict
        assert len(stats_dict[StatKey.TENSOR]) == 2  # Should have two tensors
        assert StatKey.STAT_KEY_SMOOTH_SCALE in stats_dict

    def test_get_stats_hook_with_different_shapes(self):
        """Test handling of different input shapes"""
        processor = FlexSmoothQuantProcessor(
            model=self.model,
            config=self.default_config,
            adapter=self.adapter
        )

        hook = processor.stats_collector.create_hook("test_module")
        mock_module = nn.Linear(10, 20)

        # Test different batch sizes
        for batch_size in [1, 5, 10]:
            input_tensor = (torch.randn(batch_size, 10),)
            output = torch.randn(batch_size, 20)

            hook(mock_module, input_tensor, output)

        # Verify statistics results
        stats_dict = processor.stats_collector.act_stats["test_module"]
        assert len(stats_dict[StatKey.TENSOR]) == 3  # Should have 3 tensors

        processor = FlexSmoothQuantProcessor(
            model=self.model,
            config=self.default_config,
            adapter=self.adapter
        )

        hook = processor.stats_collector.create_hook("test_module")
        mock_module = nn.Linear(10, 20)
        input_tensor = (torch.randn(5, 10),)
        output = torch.randn(5, 20)

        hook(mock_module, input_tensor, output)

        # Verify statistics results
        assert "test_module" in processor.stats_collector.act_stats
        stats_dict = processor.stats_collector.act_stats["test_module"]
        assert StatKey.TENSOR in stats_dict
        assert StatKey.STAT_KEY_SMOOTH_SCALE in stats_dict

    @patch('msmodelslim.core.api.flex_smooth_quant')
    @patch('msmodelslim.utils.logging.get_logger')
    def test_flex_smooth_quant_processor_apply_smooth(self, mock_logger, mock_flex_smooth_quant):
        """Verify _apply_smooth_to_subgraph method correctly applies smooth configuration and handles subgraph"""
        # Setup: Create subgraph object and linear module list, call _apply_smooth_to_subgraph method
        model = MockModel()
        config = FlexSmoothQuantProcessorConfig()
        adapter = MockFlexSmoothQuantInterface()
        
        processor = FlexSmoothQuantProcessor(model, config, adapter)
        
        # Create mock subgraph object and linear module list
        subgraph_obj = Mock()
        linear_modules = [model.layer1, model.layer2]
        
        # Mock _build_smooth_context method
        mock_smooth_context = Mock()
        processor._build_smooth_context = Mock(return_value=mock_smooth_context)
        
        # Call _apply_smooth_to_subgraph method
        processor._apply_smooth_to_subgraph(subgraph_obj, linear_modules)
        
        # Verify _build_smooth_context was called
        processor._build_smooth_context.assert_called_once_with(linear_modules)
        
        # Verify flex_smooth_quant was called with correct parameters
        mock_flex_smooth_quant.assert_called_once()
        call_args = mock_flex_smooth_quant.call_args
        
        # Verify parameters
        assert call_args[0][0] == subgraph_obj  # subgraph_obj
        assert call_args[0][2] == mock_smooth_context  # smooth_context
        
        # Verify FlexSmoothQuantConfig parameters
        smooth_config = call_args[0][1]
        assert smooth_config.alpha == config.alpha
        assert smooth_config.beta == config.beta

    def test_act_stats_initialization(self):
        """Test activation statistics initialization"""
        processor = FlexSmoothQuantProcessor(
            model=self.model,
            config=self.default_config,
            adapter=self.adapter
        )

        assert isinstance(processor.stats_collector.act_stats, dict)
        assert len(processor.stats_collector.act_stats) == 0

    def test_hook_handles_initialization(self):
        """Test hook handles initialization"""
        processor = FlexSmoothQuantProcessor(
            model=self.model,
            config=self.default_config,
            adapter=self.adapter
        )

        assert isinstance(processor.hook_manager.hook_handles, dict)
        assert len(processor.hook_manager.hook_handles) == 0

    def test_gradient_tracking_in_tensor(self):
        """Test tensor gradient tracking"""
        processor = FlexSmoothQuantProcessor(
            model=self.model,
            config=self.default_config,
            adapter=self.adapter
        )

        hook = processor.stats_collector.create_hook("test_module")
        mock_module = nn.Linear(10, 20)

        # Create tensor requiring gradient
        input_tensor = (torch.randn(5, 10).requires_grad_(True),)
        output = torch.randn(5, 20)

        hook(mock_module, input_tensor, output)

        # Verify statistics results are correctly obtained
        stats_dict = processor.stats_collector.act_stats["test_module"]
        assert StatKey.TENSOR in stats_dict
        assert StatKey.STAT_KEY_SMOOTH_SCALE in stats_dict

        # Verify tensor shape
        tensor_list = stats_dict[StatKey.TENSOR]
        assert len(tensor_list) == 1
        assert tensor_list[0].shape[-1] == 10  # hidden_dim

    def test_channel_max_calculation(self):
        """Test channel maximum value calculation"""
        processor = FlexSmoothQuantProcessor(
            model=self.model,
            config=self.default_config,
            adapter=self.adapter
        )

        hook = processor.stats_collector.create_hook("test_module")
        mock_module = nn.Linear(10, 20)

        # Create known tensor values to test channel max calculation
        tensor_values = torch.tensor([[1.0, -2.0, 3.0],
                                      [-1.5, 2.5, -3.5]])
        input_tensor = (tensor_values,)
        output = torch.randn(2, 20)

        hook(mock_module, input_tensor, output)

        # Verify channel maximum values
        stats_dict = processor.stats_collector.act_stats["test_module"]
        channel_max = stats_dict[StatKey.STAT_KEY_SMOOTH_SCALE]
        expected_max = torch.tensor([1.5, 2.5, 3.5])  # Max absolute value of each column

        assert torch.allclose(channel_max, expected_max)

    def test_multiple_modules_stats_accumulation(self):
        """Test statistics accumulation for multiple modules"""
        processor = FlexSmoothQuantProcessor(
            model=self.model,
            config=self.default_config,
            adapter=self.adapter
        )

        mock_module1 = nn.Linear(10, 20)
        mock_module2 = nn.Linear(10, 20)

        hook1 = processor.stats_collector.create_hook("module1")
        hook2 = processor.stats_collector.create_hook("module2")

        # Record statistics for two modules
        input_tensor1 = (torch.randn(3, 10),)
        output1 = torch.randn(3, 20)
        hook1(mock_module1, input_tensor1, output1)

        input_tensor2 = (torch.randn(4, 10),)
        output2 = torch.randn(4, 20)
        hook2(mock_module2, input_tensor2, output2)

        # Verify statistics for both modules are correctly recorded
        assert "module1" in processor.stats_collector.act_stats
        assert "module2" in processor.stats_collector.act_stats

        assert len(processor.stats_collector.act_stats["module1"][StatKey.TENSOR]) == 1
        assert len(processor.stats_collector.act_stats["module2"][StatKey.TENSOR]) == 1



class TestFlexSmoothQuantProcessorConfig:
    """FlexSmoothQuantProcessorConfig configuration test class"""

    @staticmethod
    def test_default_config():
        """Test default configuration"""
        config = FlexSmoothQuantProcessorConfig()

        assert config.type == "flex_smooth_quant"
        assert config.alpha is None
        assert config.beta is None
        assert config.enable_subgraph_type == ["norm-linear", "linear-linear", "ov", "up-down"]

    @staticmethod
    def test_custom_config():
        """Test custom configuration"""
        config = FlexSmoothQuantProcessorConfig(
            alpha=0.7,
            beta=0.9,
            enable_subgraph_type=["norm-linear", "linear-linear"]
        )

        assert config.type == "flex_smooth_quant"
        assert config.alpha == 0.7
        assert config.beta == 0.9
        assert config.enable_subgraph_type == ["norm-linear", "linear-linear"]

    @staticmethod
    def test_config_type_literal():
        """Test configuration type literal"""
        config = FlexSmoothQuantProcessorConfig()

        # Test Literal type default value
        assert isinstance(config.type, str)
        assert config.type == "flex_smooth_quant"

    @staticmethod
    def test_alpha_validation():
        """Test alpha value validation"""
        valid_config = FlexSmoothQuantProcessorConfig(alpha=0.5)
        assert valid_config.alpha == 0.5

    @staticmethod
    def test_beta_validation():
        """Test beta value validation"""      
        valid_config = FlexSmoothQuantProcessorConfig(beta=0.8)
        assert valid_config.beta == 0.8

    @staticmethod
    def test_enable_subgraph_type_validation():
        """Test subgraph type validation"""
        # Valid list
        valid_config = FlexSmoothQuantProcessorConfig(
            enable_subgraph_type=["norm-linear"]
        )
        assert valid_config.enable_subgraph_type == ["norm-linear"]

        # Multiple subgraph types
        valid_config = FlexSmoothQuantProcessorConfig(
            enable_subgraph_type=["norm-linear", "linear-linear", "ov"]
        )
        assert valid_config.enable_subgraph_type == ["norm-linear", "linear-linear", "ov"]
