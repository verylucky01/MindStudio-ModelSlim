# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
from typing import Any, List
from msmodelslim.utils.exception import SchemaValidateError, SecurityError
from msmodelslim.utils.logging import get_logger


def greater_than_zero(v: float) -> float:
    if v <= 0:
        raise SchemaValidateError("value must be greater than 0", 
                                  action="Please check the float value")
    return v


def validate_normalized_value(v: Any) -> float:
    if not isinstance(v, (float, type(None))):
        raise SchemaValidateError("value must be a float or None type",
                                  action="Please provide a float or None value")
    if v is not None and (v <= 0 or v >= 1):
        raise SchemaValidateError("value must be in the range (0, 1)",
                                  action="Please check the float value to ensure it is between 0 and 1")
    return v


def is_boolean(v: Any) -> bool:
    if not isinstance(v, bool):
        raise SchemaValidateError("value must be a boolean type", 
                                  action="Please provide a boolean value (True or False)")
    return v


def is_string_list(v: Any) -> List[str]:
    if not isinstance(v, list):
        raise SchemaValidateError("value must be a list type", 
                                  action="Please provide a list value")
    
    for item in v:
        if not isinstance(item, str):
            raise SchemaValidateError("all elements in the list must be string types", 
                                      action="Please ensure all list elements are strings")
    
    return v


def validate_str_length(input_str, str_name="string", max_len=4096):
    """
    校验输入字符串的长度是否在允许范围内

    检查字符串长度是否超过指定的最大限制，默认最大长度为4096字符。
    支持自定义字符串名称，报错信息将动态显示该名称，适配各类使用场景。

    参数:
        input_str: 需要进行长度校验的字符串
        str_name: 字符串的自定义名称，用于生成精准报错信息
        max_len: 允许的最大长度（正整数，默认4096）

    异常:
        SecurityError: 当max_len不是正整数时抛出
        SecurityError: 当input_str长度超过max_len时抛出
    """
    # 验证max_len参数合法性
    if not isinstance(max_len, int) or max_len <= 0:
        raise SecurityError("max_len must be a positive integer, current value: %r." % max_len)

    # 检查字符串长度是否超限
    if len(input_str) > max_len:
        raise SecurityError(f"The length of {str_name} should be less than {max_len}.",
                            action=f"Please make sure the {str_name} is not longer than {max_len} characters.")


def non_empty_string(v: str, field_name: str = "value") -> str:
    """
    Validate that a string is non-null and non-empty after stripping whitespace.

    Args:
        v: string to validate
        field_name: name for error message context

    Returns:
        The original string if valid

    Raises:
        SchemaValidateError: if the string is None or empty/whitespace
    """
    if v is None:
        raise SchemaValidateError(f"{field_name} must not be null",
                                  action=f"Please provide a non-empty string for {field_name}")
    if not str(v).strip():
        raise SchemaValidateError(f"{field_name} must be a non-empty string",
                                  action=f"Please provide a non-empty string for {field_name}")
    return v