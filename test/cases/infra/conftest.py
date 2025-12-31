#! /usr/bin/env python3
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
Pytest config for infra tests.

复用全局 mock 工具，避免在导入 msmodelslim 期间真正初始化配置文件和安全路径检查。
"""

from testing_utils.mock import mock_kia_library, mock_security_library, mock_init_config

# 在模块导入阶段立即生效，防止 init_config / 安全检查触发真实文件访问
mock_init_config()
mock_kia_library()
mock_security_library()
