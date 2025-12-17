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
msmodelslim.utils.exception_decorator 模块的单元测试（pytest 版）
"""

import importlib
import sys
import types

import pytest

from msmodelslim.utils.exception import (
    UnexpectedError, ModelslimError, ConfigError
)


@pytest.fixture()
def exception_decorator_mod(monkeypatch):
    """
    提供已注入假配置模块的被测模块实例，按用例隔离，避免污染其他测试。
    """
    fake_urls = types.SimpleNamespace(
        repository="https://example.com/repo",
        question_and_answer="https://example.com/faq",
    )
    fake_config_module = types.ModuleType("msmodelslim.utils.config")
    fake_config_module.msmodelslim_config = types.SimpleNamespace(urls=fake_urls)
    monkeypatch.setitem(sys.modules, "msmodelslim.utils.config", fake_config_module)

    if "msmodelslim.utils.exception_decorator" in sys.modules:
        del sys.modules["msmodelslim.utils.exception_decorator"]
    importlib.invalidate_caches()
    mod = importlib.import_module("msmodelslim.utils.exception_decorator")
    try:
        yield mod
    finally:
        if "msmodelslim.utils.exception_decorator" in sys.modules:
            del sys.modules["msmodelslim.utils.exception_decorator"]


def test_exception_handler_when_catch_specified_exception_then_convert_to_modelslim_error(exception_decorator_mod):
    """当捕获指定异常时，应转换为msModelSlim异常"""

    @exception_decorator_mod.exception_handler("配置文件格式错误", ms_err_cls=ConfigError,
                                               err_cls=ValueError,
                                               action="请检查配置文件格式")
    def test_function():
        raise ValueError("invalid literal for int() with base 10: 'abc'")

    with pytest.raises(ConfigError) as exc:
        test_function()
    assert "配置文件格式错误" in str(exc.value)
    assert "请检查配置文件格式" in str(exc.value)


def test_exception_handler_when_catch_exception_with_keyword_then_convert_only_if_keyword_matches(
        exception_decorator_mod):
    """使用关键字过滤时，仅匹配关键字的异常会被转换"""

    @exception_decorator_mod.exception_handler("文件不存在", ms_err_cls=ConfigError,
                                               err_cls=FileNotFoundError,
                                               keyword="No such file",
                                               action="请检查文件路径")
    def test_function(file_path):
        if "missing" in file_path:
            raise FileNotFoundError("No such file or directory: 'missing.txt'")
        else:
            raise FileNotFoundError("Permission denied: 'readonly.txt'")

    with pytest.raises(ConfigError) as exc:
        test_function("missing.txt")
    assert "文件不存在" in str(exc.value)

    with pytest.raises(FileNotFoundError):
        test_function("readonly.txt")


def test_exception_handler_when_modelslim_error_occurs_then_pass_through(exception_decorator_mod):
    """当发生msModelSlim异常时，应直接传递"""

    @exception_decorator_mod.exception_handler("其他错误", ms_err_cls=ConfigError)
    def test_function():
        raise ConfigError("原始配置错误", action="原始action")

    with pytest.raises(ConfigError) as exc:
        test_function()
    assert "原始配置错误" in str(exc.value)
    assert "原始action" in str(exc.value)


def test_exception_handler_when_other_exception_occurs_then_pass_through(exception_decorator_mod):
    """当发生其他类型异常时，应直接传递"""

    @exception_decorator_mod.exception_handler("配置错误", ms_err_cls=ConfigError, err_cls=ValueError)
    def test_function():
        raise TypeError("类型错误")

    with pytest.raises(TypeError):
        test_function()


def test_exception_handler_when_no_set_args_then_use_original_exception_message(exception_decorator_mod):
    """当没有设置参数时，应使用原始异常消息"""

    @exception_decorator_mod.exception_handler(err_cls=ValueError, ms_err_cls=ConfigError)
    def test_function():
        raise ValueError("原始错误消息")

    with pytest.raises(ConfigError) as exc:
        test_function()
    assert "原始错误消息" in str(exc.value)


def test_exception_handler_when_use_default_parameters_then_work_correctly(exception_decorator_mod):
    """使用默认参数时应正常工作"""

    @exception_decorator_mod.exception_handler()
    def test_function():
        raise Exception("测试异常")

    with pytest.raises(ModelslimError) as exc:
        test_function()
    assert "测试异常" in str(exc.value)


def test_exception_catcher_when_normal_execution_then_return_result(exception_decorator_mod):
    """正常执行时应返回结果"""

    @exception_decorator_mod.exception_catcher
    def test_function():
        return "success"

    assert test_function() == "success"


def test_exception_catcher_when_modelslim_error_occurs_then_log_and_re_raise(exception_decorator_mod):
    """当发生msModelSlim异常时，应记录日志并重新抛出"""

    @exception_decorator_mod.exception_catcher
    def test_function():
        raise ConfigError("配置错误", action="请检查配置")

    with pytest.raises(ConfigError) as exc:
        test_function()
    assert "配置错误" in str(exc.value)
    assert "请检查配置" in str(exc.value)


def test_exception_catcher_when_other_exception_occurs_then_convert_to_unexpected_error(exception_decorator_mod):
    """当发生其他异常时，应转换为UnexpectedError"""

    @exception_decorator_mod.exception_catcher
    def test_function():
        raise ValueError("测试异常")

    with pytest.raises(UnexpectedError) as exc:
        test_function()
    assert exception_decorator_mod.ACTION_REPORT in str(exc.value)


def test_exception_catcher_when_exception_occurs_then_preserve_original_context(exception_decorator_mod):
    """当发生异常时，应保留原始异常上下文"""

    @exception_decorator_mod.exception_catcher
    def test_function():
        raise ValueError("原始异常")

    with pytest.raises(UnexpectedError) as exc:
        test_function()
    assert exc.value.__cause__ is not None
    assert isinstance(exc.value.__cause__, ValueError)
    assert "原始异常" in str(exc.value.__cause__)


def test_exception_handler_context_when_catch_specified_exception_then_convert_to_modelslim_error(
        exception_decorator_mod):
    """上下文用法：捕获指定异常并转换为msModelSlim异常"""

    with pytest.raises(ConfigError) as exc:
        with exception_decorator_mod.exception_handler("配置文件格式错误", ms_err_cls=ConfigError,
                                                      err_cls=ValueError,
                                                      action="请检查配置文件格式"):
            raise ValueError("invalid literal for int() with base 10: 'abc'")
    assert "配置文件格式错误" in str(exc.value)
    assert "请检查配置文件格式" in str(exc.value)


def test_exception_handler_context_when_keyword_used_then_convert_only_if_matches(exception_decorator_mod):
    """上下文用法：使用关键字过滤，仅匹配时才转换"""

    # 匹配关键字，转为 ConfigError
    with pytest.raises(ConfigError):
        with exception_decorator_mod.exception_handler("文件不存在", ms_err_cls=ConfigError,
                                                      err_cls=FileNotFoundError,
                                                      keyword="No such file",
                                                      action="请检查文件路径"):
            raise FileNotFoundError("No such file or directory: 'missing.txt'")

    # 不匹配关键字，原异常透传
    with pytest.raises(FileNotFoundError):
        with exception_decorator_mod.exception_handler("文件不存在", ms_err_cls=ConfigError,
                                                      err_cls=FileNotFoundError,
                                                      keyword="No such file",
                                                      action="请检查文件路径"):
            raise FileNotFoundError("Permission denied: 'readonly.txt'")


def test_exception_handler_context_when_modelslim_error_occurs_then_pass_through(exception_decorator_mod):
    """上下文用法：发生 msModelSlim 异常时应透传"""

    with pytest.raises(ConfigError) as exc:
        with exception_decorator_mod.exception_handler("其他错误", ms_err_cls=ConfigError):
            raise ConfigError("原始配置错误", action="原始action")
    assert "原始配置错误" in str(exc.value)
    assert "原始action" in str(exc.value)


def test_exception_handler_context_when_other_exception_occurs_then_pass_through(exception_decorator_mod):
    """上下文用法：发生其他类型异常时应透传"""

    with pytest.raises(TypeError):
        with exception_decorator_mod.exception_handler("配置错误", ms_err_cls=ConfigError, err_cls=ValueError):
            raise TypeError("类型错误")


def test_exception_handler_context_when_no_set_args_then_use_original_exception_message(
        exception_decorator_mod):
    """上下文用法：未设置自定义消息时应使用原始异常消息"""

    with pytest.raises(ConfigError) as exc:
        with exception_decorator_mod.exception_handler(err_cls=ValueError, ms_err_cls=ConfigError):
            raise ValueError("原始错误消息")
    assert "原始错误消息" in str(exc.value)


def test_exception_handler_context_when_use_default_parameters_then_work_correctly(exception_decorator_mod):
    """上下文用法：使用默认参数应正常工作（Exception -> ModelslimError）"""

    with pytest.raises(ModelslimError) as exc:
        with exception_decorator_mod.exception_handler():
            raise Exception("测试异常")
    assert "测试异常" in str(exc.value)