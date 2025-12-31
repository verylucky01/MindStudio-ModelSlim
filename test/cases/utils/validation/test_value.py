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
msmodelslim.utils.validation.value 模块的单元测试
"""
from typing import Any, List
import pytest
from msmodelslim.utils.exception import SchemaValidateError
from msmodelslim.utils.validation.value import (
    greater_than_zero,
    validate_normalized_value,
    is_boolean,
    is_string_list,
    non_empty_string,
)


# ------------------------------ 测试 greater_than_zero ------------------------------
@pytest.mark.parametrize(
    "input_val, expect_return, should_raise",
    [
        (1.5, 1.5, False),
        (0.1, 0.1, False),
        (5, 5, False),
        (0.0, None, True),
        (-0.5, None, True),
        (-10, None, True),
    ],
    ids=[
        "float_1.5>0",
        "float_0.1>0",
        "int_5>0",
        "float_0.0=0",
        "float_-0.5<0",
        "int_-10<0",
    ],
)
def test_greater_than_zero(input_val: float, expect_return: float, should_raise: bool):
    """测试 greater_than_zero：验证输入是否大于 0"""
    if should_raise:
        # 捕获异常并验证错误信息、action
        with pytest.raises(SchemaValidateError) as exc_info:
            greater_than_zero(input_val)
        assert "value must be greater than 0" in str(exc_info.value)
        assert exc_info.value.action == "Please check the numeric value"
    else:
        # 验证正常返回值
        result = greater_than_zero(input_val)
        assert result == expect_return


# ------------------------------ 测试 validate_normalized_value ------------------------------
@pytest.mark.parametrize(
    "input_val, expect_return, should_raise, error_msg_keyword",
    [
        (None, None, False, ""),
        (0.5, 0.5, False, ""),
        (0.999, 0.999, False, ""),
        (0.001, 0.001, False, ""),
        ("0.5", None, True, "must be a float or None type"),
        (1, None, True, "must be a float or None type"),
        (True, None, True, "must be a float or None type"),
        (0.0, None, True, "must be in the range (0, 1)"),
        (1.0, None, True, "must be in the range (0, 1)"),
        (-0.1, None, True, "must be in the range (0, 1)"),
        (1.5, None, True, "must be in the range (0, 1)"),
    ],
    ids=[
        "input_None",
        "float_0.5_in_range",
        "float_0.999_near_1",
        "float_0.001_near_0",
        "str_0.5_type_err",
        "int_1_type_err",
        "bool_True_type_err",
        "float_0.0_out_range",
        "float_1.0_out_range",
        "float_-0.1_out_range",
        "float_1.5_out_range",
    ],
)
def test_validate_normalized_value(
    input_val: Any,
    expect_return: Any,
    should_raise: bool,
    error_msg_keyword: str,
):
    """测试 validate_normalized_value：验证输入是否为 None 或 (0,1) 区间的 float"""
    if should_raise:
        with pytest.raises(SchemaValidateError) as exc_info:
            validate_normalized_value(input_val)
        # 验证错误信息包含关键词，action 固定
        assert error_msg_keyword in str(exc_info.value)
    else:
        result = validate_normalized_value(input_val)
        assert result == expect_return


# ------------------------------ 测试 is_boolean ------------------------------
@pytest.mark.parametrize(
    "input_val, expect_return, should_raise",
    [
        (True, True, False),
        (False, False, False),
        (0, None, True),
        (1, None, True),
        ("True", None, True),
        (None, None, True),
        (1.0, None, True),
        ([], None, True),
    ],
    ids=[
        "bool_True",
        "bool_False",
        "int_0_not_bool",
        "int_1_not_bool",
        "str_True_not_bool",
        "None_not_bool",
        "float_1.0_not_bool",
        "list_empty_not_bool",
    ],
)
def test_is_boolean(input_val: Any, expect_return: bool, should_raise: bool):
    """测试 is_boolean：验证输入是否为 bool 类型"""
    if should_raise:
        with pytest.raises(SchemaValidateError) as exc_info:
            is_boolean(input_val)
        assert "value must be a boolean type" in str(exc_info.value)
        assert exc_info.value.action == "Please provide a boolean value (True or False)"
    else:
        result = is_boolean(input_val)
        assert result == expect_return


# ------------------------------ 测试 is_string_list ------------------------------
@pytest.mark.parametrize(
    "input_val, expect_return, should_raise, error_msg_keyword",
    [
        ([], [], False, ""),
        (["a", "b", "c"], ["a", "b", "c"], False, ""),
        (["123", "test", ""], ["123", "test", ""], False, ""),
        ("not a list", None, True, "must be a list type"),
        (123, None, True, "must be a list type"),
        (("a", "b"), None, True, "must be a list type"),
        (
            [1, "a", "b"],
            None,
            True,
            "all elements in the list must be string types",
        ),  # 含 int
        (
            ["a", True, "b"],
            None,
            True,
            "all elements in the list must be string types",
        ),  # 含 bool
        (
            ["a", 3.14, "b"],
            None,
            True,
            "all elements in the list must be string types",
        ),  # 含 float
        (
            ["a", None, "b"],
            None,
            True,
            "all elements in the list must be string types",
        ),  # 含 None
    ],
    ids=[
        "empty_list",
        "str_list_abc",
        "str_list_with_empty",
        "str_not_list",
        "int_not_list",
        "tuple_not_list",
        "list_with_int",
        "list_with_bool",
        "list_with_float",
        "list_with_None",
    ],
)
def test_is_string_list(
    input_val: Any,
    expect_return: List[str],
    should_raise: bool,
    error_msg_keyword: str,
):
    """测试 is_string_list：验证输入是否为列表且元素全为字符串"""
    if should_raise:
        with pytest.raises(SchemaValidateError) as exc_info:
            is_string_list(input_val)
        assert error_msg_keyword in str(exc_info.value)
        # 验证 action：根据错误类型匹配
        if "must be a list type" in error_msg_keyword:
            assert exc_info.value.action == "Please provide a list value"
        else:
            assert (
                exc_info.value.action == "Please ensure all list elements are strings"
            )
    else:
        result = is_string_list(input_val)
        assert result == expect_return
        # 额外验证返回值类型（确保是 list）
        assert isinstance(result, list)
        # 验证列表元素全为 str（非空列表时）
        if len(result) > 0:
            assert all(isinstance(item, str) for item in result)


# ------------------------------ 测试 non_empty_string ------------------------------
@pytest.mark.parametrize(
    "input_val, field_name, expect_return, should_raise, msg_keyword, action_keyword",
    [
        ("hello", "value", "hello", False, "", ""),
        ("  spaced  ", "prompt", "  spaced  ", False, "", ""),
        (None, "prompt", None, True, "prompt must not be null",
         "Please provide a non-empty string for prompt"),
        ("", "value", None, True, "value must be a non-empty string",
         "Please provide a non-empty string for value"),
        ("   ", "name", None, True, "name must be a non-empty string",
         "Please provide a non-empty string for name"),
    ],
    ids=[
        "normal_string",
        "string_with_spaces_but_not_empty",
        "none_value",
        "empty_string",
        "whitespace_only",
    ],
)
def test_non_empty_string_validation(
    input_val: Any,
    field_name: str,
    expect_return: Any,
    should_raise: bool,
    msg_keyword: str,
    action_keyword: str,
):
    """测试 non_empty_string：验证字符串非 None 且去除空白后非空。"""
    if should_raise:
        with pytest.raises(SchemaValidateError) as exc_info:
            non_empty_string(input_val, field_name=field_name)
        assert msg_keyword in str(exc_info.value)
        assert exc_info.value.action == action_keyword
    else:
        result = non_empty_string(input_val, field_name=field_name)
        # 函数应返回原始字符串（不对内容进行 strip）
        assert result == expect_return
