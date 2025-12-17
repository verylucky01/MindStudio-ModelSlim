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
import torch.nn as nn
from pydantic import ValidationError

from msmodelslim.quant.quantizer.base import QConfig
from msmodelslim.quant.quantizer.linear import LinearQConfig, LinearQuantizer
from msmodelslim.utils.exception import SpecError, SchemaValidateError


class TestLinearQConfig:
    """测试LinearQConfig配置类"""

    @staticmethod
    def test_linear_qconfig_creation():
        """测试LinearQConfig创建"""
        act_config = QConfig(
            dtype="mxfp4",
            scope="per_block",
            method="minmax",
            symmetric=True
        )
        weight_config = QConfig(
            dtype="mxfp4",
            scope="per_block",
            method="minmax",
            symmetric=True
        )

        config = LinearQConfig(act=act_config, weight=weight_config)

        assert config.act == act_config
        assert config.weight == weight_config

    @staticmethod
    def test_linear_qconfig_validation():
        """测试LinearQConfig参数验证"""
        # 测试缺失必需字段
        with pytest.raises(SchemaValidateError):  # 具体异常类型取决于pydantic配置
            LinearQConfig()


class TestLinearQuantizer:
    """测试Linear量化器"""

    def __init__(self):
        # 在__init__中初始化所有实例属性
        self.act_config = None
        self.weight_config = None
        self.config = None
        self.setup_class()  # 调用原初始化逻辑

    @staticmethod
    def test_deploy_success_with_valid_config():
        """测试deploy方法"""
        act_config = QConfig(
            dtype="mxfp4",
            scope="per_block",
            method="minmax",
            symmetric=True
        )
        weight_config = QConfig(
            dtype="mxfp4",
            scope="per_block",
            method="minmax",
            symmetric=True
        )

        config = LinearQConfig(act=act_config, weight=weight_config)
        quantizer = LinearQuantizer(config)
        quantizer.setup(nn.Linear(10, 10))
        quantizer(torch.randn((10,)))
        deployed_quantizer = quantizer.deploy()

    def setup_class(self):
        self.act_config = QConfig(
            dtype="mxfp4",
            scope="per_block",
            method="minmax",
            symmetric=True
        )
        self.weight_config = QConfig(
            dtype="mxfp4",
            scope="per_block",
            method="minmax",
            symmetric=True
        )
        self.config = LinearQConfig(act=self.act_config, weight=self.weight_config)

    def test_initialization(self):
        """测试初始化"""
        quantizer = LinearQuantizer(self.config)

    def test_setup_method_with_valid_input(self):
        """测试setup方法"""
        quantizer = LinearQuantizer(self.config)
        linear = nn.Linear(10, 20)
        quantizer.setup(linear)

    def test_setup_with_invalid_input(self):
        """测试setup方法使用无效输入"""
        quantizer = LinearQuantizer(self.config)
        with pytest.raises(ValidationError):
            quantizer.setup(None)

    def test_forward_after_setup(self):
        """测试forward方法在setup后"""
        quantizer = LinearQuantizer(self.config)
        linear = nn.Linear(10, 20)
        quantizer.setup(linear)
        result = quantizer(torch.randn(10, 10))
        assert result.shape == (10, 20)

    def test_forward_without_setup(self):
        """测试forward方法在未setup时"""
        quantizer = LinearQuantizer(self.config)
        with pytest.raises(SpecError):
            quantizer(torch.randn(10, 10))
