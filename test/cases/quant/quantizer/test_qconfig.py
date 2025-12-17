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

from msmodelslim.core.QAL.qbase import QScheme, QScope, QDType
from msmodelslim.quant.quantizer.base import QConfig
from msmodelslim.utils.exception import SchemaValidateError


class TestQConfig:
    """测试QConfig配置类"""

    def test_qconfig_creation(self):
        """测试QConfig创建"""
        config = QConfig(
            dtype="int8",
            scope="per_tensor",
            method="minmax",
            symmetric=True,
            ext={"group_size": 128}
        )

        assert config.dtype == QDType.INT8
        assert config.scope == QScope.PER_TENSOR
        assert config.method == "minmax"
        assert config.symmetric is True
        assert config.ext == {"group_size": 128}

    def test_qconfig_default_ext(self):
        """测试QConfig默认ext参数"""
        config = QConfig(
            dtype="int8",
            scope="per_tensor",
            method="minmax",
            symmetric=True
        )

        assert config.ext == {}

    def test_qconfig_to_scheme(self):
        """测试QConfig转换为QScheme"""
        config = QConfig(
            dtype="int8",
            scope="per_tensor",
            method="minmax",
            symmetric=True
        )

        scheme = config.to_scheme()
        assert isinstance(scheme, QScheme)
        assert scheme.scope == QScope.PER_TENSOR
        assert scheme.dtype == QDType.INT8
        assert scheme.symmetric is True

    def test_qconfig_to_scheme_asymmetric(self):
        """测试非对称量化配置转换"""
        config = QConfig(
            dtype="int8",
            scope="per_channel",
            method="minmax",
            symmetric=False
        )

        scheme = config.to_scheme()
        assert scheme.scope == QScope.PER_CHANNEL
        assert scheme.dtype == QDType.INT8
        assert scheme.symmetric is False

    def test_qconfig_validation_invalid_dtype(self):
        """测试无效dtype参数验证"""
        with pytest.raises(SchemaValidateError):
            QConfig.model_validate(
                dict(
                    dtype="invalid_dtype",
                    scope="per_tensor",
                    method="minmax",
                    symmetric=True
                )
            )

    def test_qconfig_validation_invalid_scope(self):
        """测试无效scope参数验证"""
        with pytest.raises(SchemaValidateError):
            QConfig(
                dtype="int8",
                scope="invalid_scope",
                method="minmax",
                symmetric=True
            )

    def test_qconfig_validation_missing_required_fields(self):
        """测试缺失必需字段验证"""
        with pytest.raises(SchemaValidateError):
            QConfig(
                dtype="int8",
                # 缺失scope
                method="minmax",
                symmetric=True
            )

    def test_qconfig_with_ext_parameters(self):
        """测试带扩展参数的配置"""
        config = QConfig(
            dtype="int8",
            scope="per_group",
            method="minmax",
            symmetric=True,
            ext={
                "group_size": 64,
                "custom_param": "value"
            }
        )

        assert config.ext["group_size"] == 64
        assert config.ext["custom_param"] == "value"
