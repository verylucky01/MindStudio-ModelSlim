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

from msmodelslim.app.quant_service.interface import BaseQuantConfig
from msmodelslim.app.quant_service.multimodal_sd_v1.quant_config import (
    DumpConfig,
    MultimodalSDConfig,
    MultimodalSDServiceConfig,
    MultimodalSDModelslimV1QuantConfig,
    load_specific_config
)
from msmodelslim.utils.exception import SchemaValidateError


def test_dump_config_default():
    """测试DumpConfig默认值"""
    config = DumpConfig()
    assert config.capture_mode == "args"
    assert config.dump_data_dir == ""


def test_dump_config_custom():
    """测试DumpConfig自定义值"""
    config = DumpConfig(capture_mode="args", dump_data_dir="/test/path")
    assert config.capture_mode == "args"
    assert config.dump_data_dir == "/test/path"


def test_multimodal_sd_config_default():
    """测试MultimodalSDConfig默认配置"""
    config = MultimodalSDConfig(dump_config=DumpConfig())
    assert isinstance(config.dump_config, DumpConfig)
    assert config.extra_params == {}


def test_multimodal_sd_config_with_extra_params():
    """测试MultimodalSDConfig包含额外参数"""
    config = MultimodalSDConfig(dump_config=DumpConfig(), extra_param1="value1", extra_param2=123)
    assert config.extra_params == {"extra_param1": "value1", "extra_param2": 123}


def test_multimodal_sd_service_config_with_dict():
    """测试MultimodalSDServiceConfig使用字典配置"""
    config_dict = {
        "dump_config": {
            "capture_mode": "args",
            "dump_data_dir": "/test"
        }
    }
    service_config = MultimodalSDServiceConfig(multimodal_sd_config=config_dict)
    assert isinstance(service_config.multimodal_sd_config, MultimodalSDConfig)
    assert service_config.multimodal_sd_config.dump_config.dump_data_dir == "/test"


def test_multimodal_sd_service_config_with_object():
    """测试MultimodalSDServiceConfig使用对象配置"""
    sd_config = MultimodalSDConfig(dump_config=DumpConfig(dump_data_dir="/test"))
    service_config = MultimodalSDServiceConfig(multimodal_sd_config=sd_config)
    assert isinstance(service_config.multimodal_sd_config, MultimodalSDConfig)
    assert service_config.multimodal_sd_config.dump_config.dump_data_dir == "/test"


def test_load_specific_config_valid():
    """测试load_specific_config加载有效配置"""
    yaml_spec = {
        "multimodal_sd_config": {
            "dump_config": {
                "capture_mode": "args",
                "dump_data_dir": "/valid/path"
            }
        }
    }
    config = load_specific_config(yaml_spec)
    assert isinstance(config, MultimodalSDServiceConfig)
    assert config.multimodal_sd_config.dump_config.dump_data_dir == "/valid/path"


def test_load_specific_config_invalid_type():
    """测试load_specific_config加载非字典配置"""
    with pytest.raises(SchemaValidateError) as excinfo:
        load_specific_config("not a dict")
    assert "task spec must be dict" in str(excinfo.value)


def test_load_specific_config_invalid_content():
    """测试load_specific_config加载无效内容配置"""
    with pytest.raises(SchemaValidateError):
        # 无效配置（缺少必要字段或类型错误）
        load_specific_config({"multimodal_sd_config": {"dump_config": 123}})


def test_multimodal_sd_modelslim_v1_quant_config_from_base():
    """测试MultimodalSDModelslimV1QuantConfig从BaseQuantConfig转换"""

    class MockBaseQuantConfig(BaseQuantConfig):
        def __init__(self):
            self.apiversion = "v1"
            self.spec = {
                "multimodal_sd_config": {
                    "dump_config": {
                        "capture_mode": "args",
                        "dump_data_dir": "/test"
                    }
                }
            }

    base_config = MockBaseQuantConfig()
    quant_config = MultimodalSDModelslimV1QuantConfig.from_base(base_config)

    assert quant_config.apiversion == "v1"
    assert isinstance(quant_config.spec, MultimodalSDServiceConfig)
    assert quant_config.spec.multimodal_sd_config.dump_config.dump_data_dir == "/test"
