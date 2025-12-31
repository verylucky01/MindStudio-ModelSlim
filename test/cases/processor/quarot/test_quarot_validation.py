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
Test validation logic for QuaRotProcessorConfig
"""

import pytest

from msmodelslim.processor.quarot.quarot import QuaRotProcessorConfig
from msmodelslim.utils.exception import SchemaValidateError


class TestQuaRotProcessorConfigValidation:
    """Test validation logic for QuaRotProcessorConfig"""

    def test_block_size_valid_values(self):
        """Test block_size with valid values"""
        # Test -1 (special value)
        config = QuaRotProcessorConfig(block_size=-1)
        assert config.block_size == -1

        # Test powers of 2
        valid_powers = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
        for power in valid_powers:
            config = QuaRotProcessorConfig(block_size=power)
            assert config.block_size == power

    def test_block_size_invalid_negative(self):
        """Test block_size with invalid negative values"""
        invalid_values = [-2, -3, -10, -100]
        for value in invalid_values:
            with pytest.raises(SchemaValidateError) as exc_info:
                QuaRotProcessorConfig(block_size=value)

            assert f"block_size must be -1 or a positive power of 2, got {value}" in str(exc_info.value)

    def test_block_size_invalid_zero(self):
        """Test block_size with zero"""
        with pytest.raises(SchemaValidateError) as exc_info:
            QuaRotProcessorConfig(block_size=0)

        assert "block_size must be -1 or a positive power of 2, got 0" in str(exc_info.value)

    def test_block_size_invalid_non_power_of_two(self):
        """Test block_size with non-power-of-two positive values"""
        invalid_values = [3, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15, 17, 18, 20, 24, 30, 33, 40, 50, 100]
        for value in invalid_values:
            with pytest.raises(SchemaValidateError) as exc_info:
                QuaRotProcessorConfig(block_size=value)

            assert f"block_size must be -1 or a positive power of 2, got {value}" in str(exc_info.value)

    def test_max_tp_size_valid_values(self):
        """Test max_tp_size with valid values"""
        valid_powers = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
        for power in valid_powers:
            config = QuaRotProcessorConfig(max_tp_size=power)
            assert config.max_tp_size == power

    def test_max_tp_size_invalid_negative(self):
        """Test max_tp_size with invalid negative values"""
        invalid_values = [-1, -2, -3, -10, -100]
        for value in invalid_values:
            with pytest.raises(SchemaValidateError) as exc_info:
                QuaRotProcessorConfig(max_tp_size=value)

            assert f"max_tp_size must be a positive power of 2 or equal to 1, got {value}" in str(exc_info.value)

    def test_max_tp_size_invalid_zero(self):
        """Test max_tp_size with zero"""
        with pytest.raises(SchemaValidateError) as exc_info:
            QuaRotProcessorConfig(max_tp_size=0)

        assert "max_tp_size must be a positive power of 2 or equal to 1, got 0" in str(exc_info.value)

    def test_max_tp_size_invalid_non_power_of_two(self):
        """Test max_tp_size with non-power-of-two values"""
        invalid_values = [3, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15, 17, 18, 20, 24, 30, 33, 40, 50, 100]
        for value in invalid_values:
            with pytest.raises(SchemaValidateError) as exc_info:
                QuaRotProcessorConfig(max_tp_size=value)

            assert f"max_tp_size must be a positive power of 2 or equal to 1, got {value}" in str(exc_info.value)

    def test_combined_valid_config(self):
        """Test combined valid configuration"""
        config = QuaRotProcessorConfig(
            block_size=64,
            max_tp_size=4,
            online=True,
            down_proj_online_layers=[0, 1, 2]
        )

        assert config.block_size == 64
        assert config.max_tp_size == 4
        assert config.online == True
        assert config.down_proj_online_layers == [0, 1, 2]

    def test_combined_invalid_config(self):
        """Test combined configuration with invalid values"""
        # Test with invalid block_size and valid max_tp_size
        with pytest.raises(SchemaValidateError) as exc_info:
            QuaRotProcessorConfig(
                block_size=3,  # Invalid: not a power of 2
                max_tp_size=4,  # Valid
                online=True
            )

        assert "block_size must be -1 or a positive power of 2, got 3" in str(exc_info.value)

        # Test with valid block_size and invalid max_tp_size
        with pytest.raises(SchemaValidateError) as exc_info:
            QuaRotProcessorConfig(
                block_size=64,  # Valid
                max_tp_size=3,  # Invalid: not a power of 2
                online=True
            )

        assert "max_tp_size must be a positive power of 2 or equal to 1, got 3" in str(exc_info.value)
