# -*- coding: utf-8 -*-
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

"""
Test validation logic for AutoroundProcessorConfig strategies field
"""

import pytest
from msmodelslim.ir.qal import QDType, QScope
from msmodelslim.core.quantizer.base import QConfig
from msmodelslim.core.quantizer.linear import LinearQConfig
from msmodelslim.processor.quant.autoround import AutoroundProcessorConfig, QuantStrategyConfig
from msmodelslim.utils.exception import SchemaValidateError


class TestAutoroundProcessorConfigValidation:
    """Test validation logic for AutoroundProcessorConfig"""

    def test_empty_strategies_raises_error(self):
        """Test that empty strategies list raises SchemaValidateError"""
        with pytest.raises(SchemaValidateError) as exc_info:
            AutoroundProcessorConfig(strategies=[])
        
        assert "strategies field cannot be empty" in str(exc_info.value)
        assert "at least one quantization strategy must be configured" in str(exc_info.value)

    def test_default_strategies_raises_error(self):
        """Test that default empty strategies list raises SchemaValidateError"""
        with pytest.raises(SchemaValidateError) as exc_info:
            AutoroundProcessorConfig()
        
        assert "strategies field cannot be empty" in str(exc_info.value)

    def test_per_group_with_valid_group_size_passes(self):
        """Test that per_group scope with valid group_size passes validation"""
        weight_config = QConfig(
            dtype=QDType.INT8,
            scope=QScope.PER_GROUP,
            symmetric=True,
            method="minmax",
            ext={"group_size": 128}
        )
        
        act_config = QConfig(
            dtype=QDType.INT8,
            scope=QScope.PER_TENSOR,
            symmetric=True,
            method="minmax",
            ext={}
        )
        
        qconfig = LinearQConfig(act=act_config, weight=weight_config)
        strategy = QuantStrategyConfig(qconfig=qconfig, include=["*"], exclude=[])
        
        # Should not raise exception
        config = AutoroundProcessorConfig(strategies=[strategy])
        assert len(config.strategies) == 1

    def test_per_group_without_group_size_raises_error(self):
        """Test that per_group scope without group_size raises SchemaValidateError"""
        weight_config = QConfig(
            dtype=QDType.INT8,
            scope=QScope.PER_GROUP,
            symmetric=True,
            method="minmax",
            ext={}
        )
        
        act_config = QConfig(
            dtype=QDType.INT8,
            scope=QScope.PER_TENSOR,
            symmetric=True,
            method="minmax",
            ext={}
        )
        
        qconfig = LinearQConfig(act=act_config, weight=weight_config)
        strategy = QuantStrategyConfig(qconfig=qconfig, include=["*"], exclude=[])
        
        with pytest.raises(SchemaValidateError) as exc_info:
            AutoroundProcessorConfig(strategies=[strategy])
        
        assert "must contain group_size" in str(exc_info.value)
        assert "strategies[0].qconfig.weight" in str(exc_info.value)

    def test_per_group_with_invalid_group_size_raises_error(self):
        """Test that per_group scope with invalid group_size raises SchemaValidateError"""
        weight_config = QConfig(
            dtype=QDType.INT8,
            scope=QScope.PER_GROUP,
            symmetric=True,
            method="minmax",
            ext={"group_size": 0}
        )
        
        act_config = QConfig(
            dtype=QDType.INT8,
            scope=QScope.PER_TENSOR,
            symmetric=True,
            method="minmax",
            ext={}
        )
        
        qconfig = LinearQConfig(act=act_config, weight=weight_config)
        strategy = QuantStrategyConfig(qconfig=qconfig, include=["*"], exclude=[])
        
        with pytest.raises(SchemaValidateError) as exc_info:
            AutoroundProcessorConfig(strategies=[strategy])
        
        assert "must be a positive integer" in str(exc_info.value)
        assert "group_size=0" in str(exc_info.value)

    def test_per_group_with_non_integer_group_size_raises_error(self):
        """Test that per_group scope with non-integer group_size raises SchemaValidateError"""
        weight_config = QConfig(
            dtype=QDType.INT8,
            scope=QScope.PER_GROUP,
            symmetric=True,
            method="minmax",
            ext={"group_size": "128"}
        )
        
        act_config = QConfig(
            dtype=QDType.INT8,
            scope=QScope.PER_TENSOR,
            symmetric=True,
            method="minmax",
            ext={}
        )
        
        qconfig = LinearQConfig(act=act_config, weight=weight_config)
        strategy = QuantStrategyConfig(qconfig=qconfig, include=["*"], exclude=[])
        
        with pytest.raises(SchemaValidateError) as exc_info:
            AutoroundProcessorConfig(strategies=[strategy])
        
        assert "must be a positive integer" in str(exc_info.value)

    def test_non_per_group_with_group_size_raises_error(self):
        """Test that non-per_group scope with group_size raises SchemaValidateError"""
        weight_config = QConfig(
            dtype=QDType.INT8,
            scope=QScope.PER_CHANNEL,
            symmetric=True,
            method="minmax",
            ext={"group_size": 128}
        )
        
        act_config = QConfig(
            dtype=QDType.INT8,
            scope=QScope.PER_TENSOR,
            symmetric=True,
            method="minmax",
            ext={}
        )
        
        qconfig = LinearQConfig(act=act_config, weight=weight_config)
        strategy = QuantStrategyConfig(qconfig=qconfig, include=["*"], exclude=[])
        
        with pytest.raises(SchemaValidateError) as exc_info:
            AutoroundProcessorConfig(strategies=[strategy])
        
        assert "should not contain group_size" in str(exc_info.value)
        assert "strategies[0].qconfig.weight" in str(exc_info.value)

    def test_non_per_group_without_group_size_passes(self):
        """Test that non-per_group scope without group_size passes validation"""
        weight_config = QConfig(
            dtype=QDType.INT8,
            scope=QScope.PER_CHANNEL,
            symmetric=True,
            method="minmax",
            ext={}
        )
        
        act_config = QConfig(
            dtype=QDType.INT8,
            scope=QScope.PER_TENSOR,
            symmetric=True,
            method="minmax",
            ext={}
        )
        
        qconfig = LinearQConfig(act=act_config, weight=weight_config)
        strategy = QuantStrategyConfig(qconfig=qconfig, include=["*"], exclude=[])
        
        # Should not raise exception
        config = AutoroundProcessorConfig(strategies=[strategy])
        assert len(config.strategies) == 1

    def test_multiple_strategies_validation(self):
        """Test validation with multiple strategies"""
        # First strategy: per_group scope with valid group_size
        weight_config1 = QConfig(
            dtype=QDType.INT8,
            scope=QScope.PER_GROUP,
            symmetric=True,
            method="minmax",
            ext={"group_size": 64}
        )
        act_config1 = QConfig(
            dtype=QDType.INT8,
            scope=QScope.PER_TENSOR,
            symmetric=True,
            method="minmax",
            ext={}
        )
        strategy1 = QuantStrategyConfig(
            qconfig=LinearQConfig(act=act_config1, weight=weight_config1),
            include=["*"],
            exclude=[]
        )
        
        # Second strategy: per_channel scope without group_size
        weight_config2 = QConfig(
            dtype=QDType.INT8,
            scope=QScope.PER_CHANNEL,
            symmetric=True,
            method="minmax",
            ext={}
        )
        act_config2 = QConfig(
            dtype=QDType.INT8,
            scope=QScope.PER_TENSOR,
            symmetric=True,
            method="minmax",
            ext={}
        )
        strategy2 = QuantStrategyConfig(
            qconfig=LinearQConfig(act=act_config2, weight=weight_config2),
            include=["*"],
            exclude=[]
        )
        
        # Should not raise exception
        config = AutoroundProcessorConfig(strategies=[strategy1, strategy2])
        assert len(config.strategies) == 2

    def test_activation_per_group_validation(self):
        """Test validation for activation configuration with per_group scope"""
        weight_config = QConfig(
            dtype=QDType.INT8,
            scope=QScope.PER_CHANNEL,
            symmetric=True,
            method="minmax",
            ext={}
        )
        
        # Activation with per_group scope should have group_size
        act_config = QConfig(
            dtype=QDType.INT8,
            scope=QScope.PER_GROUP,
            symmetric=True,
            method="minmax",
            ext={"group_size": 32}
        )
        
        qconfig = LinearQConfig(act=act_config, weight=weight_config)
        strategy = QuantStrategyConfig(qconfig=qconfig, include=["*"], exclude=[])
        
        # Should not raise exception
        config = AutoroundProcessorConfig(strategies=[strategy])
        assert len(config.strategies) == 1

    def test_activation_per_group_without_group_size_raises_error(self):
        """Test that activation per_group scope without group_size raises error"""
        weight_config = QConfig(
            dtype=QDType.INT8,
            scope=QScope.PER_CHANNEL,
            symmetric=True,
            method="minmax",
            ext={}
        )
        
        # Activation with per_group scope but no group_size
        act_config = QConfig(
            dtype=QDType.INT8,
            scope=QScope.PER_GROUP,
            symmetric=True,
            method="minmax",
            ext={}
        )
        
        qconfig = LinearQConfig(act=act_config, weight=weight_config)
        strategy = QuantStrategyConfig(qconfig=qconfig, include=["*"], exclude=[])
        
        with pytest.raises(SchemaValidateError) as exc_info:
            AutoroundProcessorConfig(strategies=[strategy])
        
        assert "must contain group_size" in str(exc_info.value)
        assert "strategies[0].qconfig.act" in str(exc_info.value)

    def test_parse_scale_dtype_valid_types(self):
        """Test _parse_scale_dtype with valid scale dtypes"""
        from msmodelslim.processor.quant.autoround import _parse_scale_dtype
        import torch
        
        # Test valid scale dtypes
        assert _parse_scale_dtype("float16") == torch.float16
        assert _parse_scale_dtype("float32") == torch.float32
        assert _parse_scale_dtype("bfloat16") == torch.bfloat16

    def test_parse_scale_dtype_invalid_type(self):
        """Test _parse_scale_dtype with invalid scale dtype"""
        from msmodelslim.processor.quant.autoround import _parse_scale_dtype
        
        # Test invalid scale dtype
        with pytest.raises(SchemaValidateError) as exc_info:
            _parse_scale_dtype("float64")
        
        assert "Unsupported scale dtype 'float64'" in str(exc_info.value)
        assert "supported types are:" in str(exc_info.value)
        assert "float16" in str(exc_info.value)
        assert "float32" in str(exc_info.value)
        assert "bfloat16" in str(exc_info.value)

    def test_parse_scale_dtype_unsupported_type(self):
        """Test _parse_scale_dtype with unsupported scale dtype"""
        from msmodelslim.processor.quant.autoround import _parse_scale_dtype
        
        # Test another unsupported scale dtype
        with pytest.raises(SchemaValidateError) as exc_info:
            _parse_scale_dtype("int8")
        
        assert "Unsupported scale dtype 'int8'" in str(exc_info.value)
        assert "supported types are:" in str(exc_info.value)

    def test_validate_strategies_with_invalid_scale_dtype(self):
        """Test validate_strategies catches and re-raises exceptions with position info"""
        # Create a strategy with invalid scale dtype
        weight_config = QConfig(
            dtype=QDType.INT8,
            scope=QScope.PER_CHANNEL,
            symmetric=True,
            method="minmax",
            ext={"scale_dtype": "float64"}  # Invalid scale dtype
        )
        act_config = QConfig(
            dtype=QDType.INT8,
            scope=QScope.PER_TENSOR,
            symmetric=True,
            method="minmax",
            ext={}
        )
        qconfig = LinearQConfig(act=act_config, weight=weight_config)
        strategy = QuantStrategyConfig(qconfig=qconfig, include=["*"], exclude=[])
        
        with pytest.raises(SchemaValidateError) as exc_info:
            AutoroundProcessorConfig(strategies=[strategy])
        
        assert "Configuration validation failed for strategies[0]" in str(exc_info.value)
        assert "Unsupported scale dtype 'float64'" in str(exc_info.value)
        assert "Please check the configuration of strategies[0].qconfig" in str(exc_info.value)

    def test_validate_strategies_with_multiple_strategies_error_position(self):
        """Test validate_strategies with multiple strategies shows correct error position"""
        # First strategy - valid
        weight_config1 = QConfig(
            dtype=QDType.INT8,
            scope=QScope.PER_CHANNEL,
            symmetric=True,
            method="minmax",
            ext={}
        )
        act_config1 = QConfig(
            dtype=QDType.INT8,
            scope=QScope.PER_TENSOR,
            symmetric=True,
            method="minmax",
            ext={}
        )
        qconfig1 = LinearQConfig(act=act_config1, weight=weight_config1)
        strategy1 = QuantStrategyConfig(qconfig=qconfig1, include=["*"], exclude=[])
        
        # Second strategy - invalid scale dtype
        weight_config2 = QConfig(
            dtype=QDType.INT8,
            scope=QScope.PER_CHANNEL,
            symmetric=True,
            method="minmax",
            ext={"scale_dtype": "int8"}  # Invalid scale dtype
        )
        act_config2 = QConfig(
            dtype=QDType.INT8,
            scope=QScope.PER_TENSOR,
            symmetric=True,
            method="minmax",
            ext={}
        )
        qconfig2 = LinearQConfig(act=act_config2, weight=weight_config2)
        strategy2 = QuantStrategyConfig(qconfig=qconfig2, include=["*"], exclude=[])
        
        with pytest.raises(SchemaValidateError) as exc_info:
            AutoroundProcessorConfig(strategies=[strategy1, strategy2])
        
        assert "Configuration validation failed for strategies[1]" in str(exc_info.value)
        assert "Unsupported scale dtype 'int8'" in str(exc_info.value)
        assert "Please check the configuration of strategies[1].qconfig" in str(exc_info.value)
