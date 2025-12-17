#  -*- coding: utf-8 -*-
#  Copyright (c) 2025-2025 Huawei Technologies Co., Ltd.
#  #
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
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
from unittest.mock import Mock, patch
from testing_utils.mock import mock_kia_library, mock_security_library, mock_init_config

# Mock必要的库
mock_init_config()
mock_kia_library()
mock_security_library()


# 创建通用的mock对象
@pytest.fixture
def mock_torch():
    """Mock torch库"""
    with patch('torch') as mock_torch:
        mock_torch.device.return_value = Mock()
        mock_torch.manual_seed.return_value = None
        mock_torch.npu.manual_seed.return_value = None
        mock_torch.npu.manual_seed_all.return_value = None
        mock_torch.npu.Stream.return_value = Mock()
        yield mock_torch


@pytest.fixture
def mock_wan():
    """Mock wan库"""
    with patch('wan') as mock_wan:
        mock_wan.WanT2V.return_value = Mock()
        yield mock_wan


@pytest.fixture
def mock_wan_configs():
    """Mock wan配置"""
    with patch('wan.configs') as mock_configs:
        mock_configs.WAN_CONFIGS = {
            "t2v-14B": Mock(),
            "t2i-14B": Mock(),
            "i2v-14B": Mock()
        }
        mock_configs.SIZE_CONFIGS = {
            "1280*720": (1280, 720),
            "832*480": (832, 480),
            "480*832": (480, 832)
        }
        mock_configs.SUPPORTED_SIZES = {
            "t2v-14B": ["1280*720", "832*480"],
            "t2i-14B": ["1280*720", "832*480"],
            "i2v-14B": ["1280*720", "832*480", "480*832"]
        }
        yield mock_configs


@pytest.fixture
def mock_mindiesd():
    """Mock mindiesd库"""
    with patch('mindiesd') as mock_mindiesd:
        mock_mindiesd.CacheConfig.return_value = Mock()
        mock_mindiesd.CacheAgent.return_value = Mock()
        yield mock_mindiesd


@pytest.fixture
def mock_tqdm():
    """Mock tqdm库"""
    with patch('tqdm.tqdm') as mock_tqdm:
        mock_tqdm.return_value = [None]
        yield mock_tqdm
