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

"""
Pytest config for model tests.

复用全局 mock 工具，避免在导入 msmodelslim 期间真正初始化配置文件和安全路径检查。
同时通过 fixture 方式按会话级别 mock 缺失的第三方依赖（如 wcmatch），
避免对其他测试任务造成长期污染。
"""

import sys
from unittest.mock import MagicMock

import pytest

from testing_utils.mock import mock_security_library, mock_kia_library, mock_init_config


mock_init_config()
mock_kia_library()
mock_security_library()

# 记录 wcmatch 的原始状态，用于 pytest_unconfigure 中清理
_wcmatch_original = None
_wcmatch_mock_used = False

try:
    import wcmatch
    _wcmatch_original = sys.modules.get("wcmatch")
except ImportError:
    # 如果导入失败，创建 mock
    _wcmatch_mock_used = True
    _wcmatch_original = None
    sys.modules["wcmatch"] = MagicMock()
    sys.modules["wcmatch"].fnmatch = MagicMock()


def pytest_unconfigure(config):
    """
    在测试结束后清理 wcmatch mock，恢复原始模块（如果存在）。
    仅清理在本 conftest 中新创建的 mock，避免影响外部真实依赖。
    """
    global _wcmatch_original, _wcmatch_mock_used
    
    if _wcmatch_mock_used:
        if "wcmatch" in sys.modules:
            del sys.modules["wcmatch"]
        if _wcmatch_original is not None:
            sys.modules["wcmatch"] = _wcmatch_original
