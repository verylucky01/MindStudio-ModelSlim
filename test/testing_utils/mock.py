#  -*- coding: utf-8 -*-
#  Copyright (c) 2024-2024 Huawei Technologies Co., Ltd.
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


import json
import os
import stat
import sys
import yaml

from unittest.mock import MagicMock


def _mock_json_safe_dump(obj, path, indent=None, extensions="json", check_user_stat=True):
    default_mode = stat.S_IWUSR | stat.S_IRUSR  # 600
    with os.fdopen(os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, mode=default_mode), "w") as json_file:
        json.dump(obj, json_file, indent=indent)


def _mock_get_valid_write_path(path: str, *args, **kwarg) -> str:
    return path


def _mock_yaml_safe_load(path, *args, **kwargs):
    """Mock yaml_safe_load function that reads yaml file from path and converts to dict"""
    with open(path, 'r', encoding='utf-8') as file:
        return yaml.safe_load(file)


def _mock_json_safe_load(path, *args, **kwargs):
    """Mock json_safe_load function that reads json file from path and converts to dict"""
    with open(path, 'r', encoding='utf-8') as file:
        return json.load(file)


def mock_kia_library():
    sys.modules['msmodelslim.pytorch.llm_ptq.anti_outlier.anti_utils'] = MagicMock()
    sys.modules['msmodelslim.pytorch.llm_ptq.llm_ptq_tools.quant_funcs'] = MagicMock()
    sys.modules['msmodelslim.pytorch.llm_sparsequant.atomic_power_outlier'] = MagicMock()
    sys.modules['msmodelslim.pytorch.lowbit.atomic_power_outlier'] = MagicMock()
    sys.modules['msmodelslim.pytorch.lowbit.calibration'] = MagicMock()
    sys.modules['msmodelslim.pytorch.lowbit.quant_modules'] = MagicMock()


def mock_security_library():
    sys.modules['msmodelslim.utils.security.path'] = MagicMock()
    sys.modules['msmodelslim.utils.security.path'].json_safe_dump = _mock_json_safe_dump
    sys.modules['msmodelslim.utils.security.path'].json_safe_load = _mock_json_safe_load
    sys.modules['msmodelslim.utils.security.path'].yaml_safe_load = _mock_yaml_safe_load
    sys.modules['msmodelslim.utils.security.path'].get_valid_write_path = _mock_get_valid_write_path
    sys.modules['msmodelslim.utils.security.path'].get_valid_path = _mock_get_valid_write_path
    sys.modules['msmodelslim.utils.security.path'].get_valid_read_path = _mock_get_valid_write_path
    sys.modules['msmodelslim.utils.security.path'].get_write_directory = _mock_get_valid_write_path

    sys.modules['ascend_utils.common.security.path'] = MagicMock()
    sys.modules['ascend_utils.common.security.path'].json_safe_dump = _mock_json_safe_dump
    sys.modules['ascend_utils.common.security.path'].json_safe_load = _mock_json_safe_load
    sys.modules['ascend_utils.common.security.path'].yaml_safe_load = _mock_yaml_safe_load
    sys.modules['ascend_utils.common.security.path'].get_valid_write_path = _mock_get_valid_write_path
    sys.modules['ascend_utils.common.security.path'].get_valid_path = _mock_get_valid_write_path
    sys.modules['ascend_utils.common.security.path'].get_write_directory = _mock_get_valid_write_path


def mock_init_config():
    """Mock init_config function and related config modules"""
    # Mock the config module
    config_mock = MagicMock()

    # Create a mock config object that mimics the structure of ModelSlimConfig
    mock_config = MagicMock()
    mock_config.urls.repository = "mocked_url"
    mock_config.env_vars.log_level = "info"
    mock_config.env_vars.custom_practice_repo = None

    # Mock the init_config function to return the mock config
    config_mock.init_config.return_value = mock_config
    config_mock.msmodelslim_config = mock_config

    # Mock the config classes
    config_mock.ModelSlimConfig = MagicMock()
    config_mock.URLs = MagicMock()
    config_mock.EnvVars = MagicMock()

    # Register the mock module
    sys.modules['msmodelslim.utils.config'] = config_mock
