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
from pydantic import ValidationError

from msmodelslim.quant.quantizer.base import AutoActQuantizer, AutoWeightQuantizer, QConfig
from msmodelslim.utils.exception import SchemaValidateError


class TestAutoActQuantizerFactory:
    """测试AutoActQuantizer工厂构造方法"""

    def test_from_config_with_invalid_config(self):
        """测试from_config工厂方法使用无效配置"""
        # 测试None配置
        with pytest.raises(ValidationError):
            AutoActQuantizer.from_config(None)

        # 测试非QConfig对象
        with pytest.raises(SchemaValidateError):
            AutoActQuantizer.from_config({"dtype": "int8"})

    def test_register_new_subclass_and_can_create_instance_using_from_config(self):
        from msmodelslim.core.QAL.qregistry import QABCRegistry

        test_scheme = QConfig(dtype="int8", scope="per_channel", method="test", symmetric=True).to_scheme()

        @QABCRegistry.register(dispatch_key=(test_scheme, "test"), abc_class=AutoActQuantizer)
        class MyActQuantizer(AutoActQuantizer):

            def __init__(self, config: QConfig):
                super().__init__()

        config = QConfig(dtype="int8", scope="per_channel", method="test", symmetric=True)
        quantizer = AutoActQuantizer.from_config(config)
        assert isinstance(quantizer, MyActQuantizer)


class TestAutoWeightQuantizerFactory:
    """测试AutoWeightQuantizer工厂构造方法"""

    def test_from_config_with_invalid_config(self):
        """测试from_config工厂方法使用无效配置"""
        # 测试None配置
        with pytest.raises(ValidationError):
            AutoWeightQuantizer.from_config(None)

        # 测试非QConfig对象
        with pytest.raises(SchemaValidateError):
            AutoWeightQuantizer.from_config({"dtype": "int8"})

    def test_register_new_subclass_and_can_create_instance_using_from_config(self):
        from msmodelslim.core.QAL.qregistry import QABCRegistry

        test_scheme = QConfig(dtype="int8", scope="per_channel", method="test", symmetric=True).to_scheme()

        @QABCRegistry.register(dispatch_key=(test_scheme, "test"), abc_class=AutoWeightQuantizer)
        class MyWeightQuantizer(AutoWeightQuantizer):

            def __init__(self, config: QConfig):
                super().__init__()

        config = QConfig(dtype="int8", scope="per_channel", method="test", symmetric=True)
        quantizer = AutoWeightQuantizer.from_config(config)
        assert isinstance(quantizer, MyWeightQuantizer)
