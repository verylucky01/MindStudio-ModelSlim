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
msmodelslim.utils.validation.conversion 模块的单元测试（pytest 版）
"""

from pathlib import Path
from unittest.mock import patch

import pytest

from msmodelslim.utils.exception import SchemaValidateError, UnsupportedError
from msmodelslim.utils.validation.conversion import (
    convert_to_bool,
    convert_to_readable_dir,
    convert_to_writable_dir,
    convert_to_readable_file
)


class TestConvertToBool:
    """测试 convert_to_bool 函数"""

    def test_convert_to_bool_when_input_is_true_string_then_return_true(self):
        """当输入为 'True' 字符串时，应返回 True"""
        result = convert_to_bool("True")
        assert result is True

    def test_convert_to_bool_when_input_is_false_string_then_return_false(self):
        """当输入为 'False' 字符串时，应返回 False"""
        result = convert_to_bool("False")
        assert result is False

    def test_convert_to_bool_when_input_is_other_string_then_raise_schema_validate_error(self):
        """当输入为其他字符串时，应抛出 SchemaValidateError"""
        with pytest.raises(SchemaValidateError) as exc:
            convert_to_bool("yes")
        assert "yes is not True or False" in str(exc.value)
        assert "Please ensure the input is literally True or False" in str(exc.value)


class TestConvertToReadableDir:
    """测试 convert_to_readable_dir 函数"""

    @patch('msmodelslim.utils.validation.conversion.get_valid_read_path')
    def test_convert_to_readable_dir_when_input_is_valid_string_then_return_path(self, mock_get_valid_read_path):
        """当输入为有效字符串时，应返回 Path 对象"""
        mock_get_valid_read_path.return_value = "/test/dir"
        result = convert_to_readable_dir("/test/dir")

        assert isinstance(result, Path)
        assert result.as_posix().endswith("test/dir")
        mock_get_valid_read_path.assert_called_once_with("/test/dir", is_dir=True)

    def test_convert_to_readable_dir_when_input_is_not_string_then_raise_unsupported_error(self):
        """当输入不是字符串时，应抛出 UnsupportedError"""
        with pytest.raises(UnsupportedError) as exc:
            convert_to_readable_dir(123)
        assert "Unsupported type converted to readable dir: <class 'int'>" in str(exc.value)
        assert "Please ensure the input is a string" in str(exc.value)


class TestConvertToWritableDir:
    """测试 convert_to_writable_dir 函数"""

    @patch('msmodelslim.utils.validation.conversion.get_write_directory')
    def test_convert_to_writable_dir_when_input_is_valid_string_then_return_path(self, mock_get_write_directory):
        """当输入为有效字符串时，应返回 Path 对象"""
        mock_get_write_directory.return_value = "/test/dir"
        result = convert_to_writable_dir("/test/dir")

        assert isinstance(result, Path)
        assert result.as_posix().endswith("test/dir")
        mock_get_write_directory.assert_called_once_with("/test/dir", write_mode=0o750)

    def test_convert_to_writable_dir_when_input_is_not_string_then_raise_unsupported_error(self):
        """当输入不是字符串时，应抛出 UnsupportedError"""
        with pytest.raises(UnsupportedError) as exc:
            convert_to_writable_dir(456)
        assert "Unsupported type converted to writable dir: <class 'int'>" in str(exc.value)
        assert "Please ensure the input is a string" in str(exc.value)


class TestConvertToReadableFile:
    """测试 convert_to_readable_file 函数"""

    @patch('msmodelslim.utils.validation.conversion.get_valid_read_path')
    def test_convert_to_readable_file_when_input_is_valid_string_then_return_path(self, mock_get_valid_read_path):
        """当输入为有效字符串时，应返回 Path 对象"""
        mock_get_valid_read_path.return_value = "/test/file.txt"
        result = convert_to_readable_file("/test/file.txt")

        assert isinstance(result, Path)
        assert result.as_posix().endswith("test/file.txt")
        mock_get_valid_read_path.assert_called_once_with("/test/file.txt", is_dir=False)

    def test_convert_to_readable_file_when_input_is_not_string_then_raise_unsupported_error(self):
        """当输入不是字符串时，应抛出 UnsupportedError"""
        with pytest.raises(UnsupportedError) as exc:
            convert_to_readable_file(789)
        assert "Unsupported type converted to readable file: <class 'int'>" in str(exc.value)
        assert "Please ensure the input is a string" in str(exc.value)

    def test_convert_to_readable_file_when_input_is_dict_then_raise_unsupported_error(self):
        """当输入为字典时，应抛出 UnsupportedError"""
        with pytest.raises(UnsupportedError) as exc:
            convert_to_readable_file({"path": "/test/file.txt"})
        assert "Unsupported type converted to readable file: <class 'dict'>" in str(exc.value)


class TestConversionIntegration:
    """转换函数的集成测试"""

    def test_convert_to_bool_when_input_not_literal_true_false_then_raise_schema_validate_error(self):
        """测试convert_to_bool：当输入非字面True/False时，应抛出SchemaValidateError"""
        # 条件：大小写不同或非严格匹配
        with pytest.raises(SchemaValidateError):
            convert_to_bool("true")

        with pytest.raises(SchemaValidateError):
            convert_to_bool("false")

        with pytest.raises(SchemaValidateError):
            convert_to_bool("TRUE")

        with pytest.raises(SchemaValidateError):
            convert_to_bool("FALSE")

    @patch('msmodelslim.utils.validation.conversion.get_valid_read_path')
    def test_convert_to_readable_dir_when_security_validation_fails_then_raise_exception(self, mock_get_valid_read_path):
        """测试convert_to_readable_dir：当安全校验失败时，应抛出异常"""
        mock_get_valid_read_path.side_effect = Exception("Security validation failed")

        with pytest.raises(Exception):
            convert_to_readable_dir("/invalid/path")

    @patch('msmodelslim.utils.validation.conversion.get_write_directory')
    def test_convert_to_writable_dir_when_directory_missing_then_create_and_return_path(self, mock_get_write_directory):
        """测试convert_to_writable_dir：当目录缺失时，应创建并返回路径"""
        mock_get_write_directory.return_value = "/new/dir"
        result = convert_to_writable_dir("/new/dir")

        assert result.as_posix().endswith("new/dir")
        mock_get_write_directory.assert_called_once()

    @patch('msmodelslim.utils.validation.conversion.get_valid_read_path')
    def test_convert_to_readable_file_when_validation_pass_then_return_path(self, mock_get_valid_read_path):
        """测试convert_to_readable_file：当校验通过时，应返回路径"""
        mock_get_valid_read_path.return_value = "/existing/file.txt"
        result = convert_to_readable_file("/existing/file.txt")

        assert result.as_posix().endswith("existing/file.txt")
        mock_get_valid_read_path.assert_called_once_with("/existing/file.txt", is_dir=False)
