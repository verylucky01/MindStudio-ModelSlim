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
from unittest.mock import MagicMock, patch

import pytest


from .base import invoke_analysis_test


@pytest.mark.smoke
def test_analysis_cli_coverage():
    """测试分析模块CLI主函数代码覆盖"""
    # 测试基本功能
    result = invoke_analysis_test(
        metrics="kurtosis",
        patterns=["*"],
        topk=15
    )

    # 测试不同算法
    for metrics in ["kurtosis", "std", "quantile"]:
        result = invoke_analysis_test(
            metrics=metrics,
            patterns=["*"],
            topk=10
        )

    # 测试不同层模式
    for patterns in [["*"], ["*attention*"], ["*mlp*"], ["*attention*", "*mlp*"]]:
        result = invoke_analysis_test(
            metrics="kurtosis",
            patterns=patterns,
            topk=15
        )

    # 测试不同topk值
    for topk in [5, 10, 15, 20]:
        result = invoke_analysis_test(
            metrics="kurtosis",
            patterns=["*"],
            topk=topk
        )

    # 测试trust_remote_code（现在已固定为False）
    result = invoke_analysis_test(
        metrics="kurtosis",
        patterns=["*"],
        topk=15
    )
