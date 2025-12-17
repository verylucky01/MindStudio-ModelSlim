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
msmodelslim.utils.yaml_database 模块的单元测试
"""
from pathlib import Path
import yaml
import pytest

from msmodelslim.utils.exception import (
    SchemaValidateError,
    SecurityError,
    UnsupportedError,
)
from msmodelslim.utils.yaml_database import YamlDatabase


# 辅助函数：创建测试用YAML文件
def create_test_yaml(tmp_path: Path, filename: str, content: dict):
    file_path = tmp_path / f"{filename}.yaml"
    with open(file_path, "w") as f:
        yaml.safe_dump(content, f)


# ------------------------------ 测试迭代器方法 ------------------------------
def test_yaml_database_iter(tmp_path: Path):
    """测试迭代器功能"""
    create_test_yaml(tmp_path, "test1", {"key": "value1"})
    create_test_yaml(tmp_path, "test2", {"key": "value2"})
    (tmp_path / "test3.txt").write_text("not a yaml file")

    db = YamlDatabase(config_dir=tmp_path)

    items = list(db)

    assert len(items) == 2
    assert "test1" in items
    assert "test2" in items
    assert "test3" not in items


def test_yaml_database_iter_empty(tmp_path: Path):
    """测试空目录的迭代器"""
    db = YamlDatabase(config_dir=tmp_path)
    assert len(list(db)) == 0


def test_yaml_database_getitem_non_existing(tmp_path: Path):
    """测试获取不存在的键"""
    db = YamlDatabase(config_dir=tmp_path)

    with pytest.raises(UnsupportedError) as exc_info:
        _ = db["nonexistent"]

    assert "yaml database key nonexistent not found" in str(exc_info.value)
    assert exc_info.value.action == "Please check the yaml database"


def test_yaml_database_getitem_invalid_key_type(tmp_path: Path):
    """测试使用非字符串作为键"""
    db = YamlDatabase(config_dir=tmp_path)

    with pytest.raises(SchemaValidateError) as exc_info:
        _ = db[123]
    assert exc_info.value.action == "Please make sure the key is a string"


def test_yaml_database_setitem_read_only(tmp_path: Path):
    """测试在只读模式下尝试设置键值"""
    db = YamlDatabase(config_dir=tmp_path, read_only=True)

    with pytest.raises(SecurityError) as exc_info:
        db["new_key"] = {"new": "value"}

    assert f"yaml database {tmp_path} is read-only" in str(exc_info.value)
    assert exc_info.value.action == "Writing operation is forbidden"


def test_yaml_database_setitem_invalid_key_type(tmp_path: Path):
    """测试使用非字符串作为设置键"""
    db = YamlDatabase(config_dir=tmp_path, read_only=False)

    with pytest.raises(SchemaValidateError) as exc_info:
        db[123] = {"key": "value"}  # 使用整数作为键

    assert exc_info.value.action == "Please make sure the key is a string"


# ------------------------------ 测试__contains__方法 ------------------------------
def test_yaml_database_contains_existing(tmp_path: Path):
    """测试检查存在的键"""
    create_test_yaml(tmp_path, "existing", {"key": "value"})

    db = YamlDatabase(config_dir=tmp_path)

    assert "existing" in db
    assert "nonexistent" not in db


# ------------------------------ 测试values()方法 ------------------------------
def test_yaml_database_values(tmp_path: Path):
    """测试获取所有值"""
    create_test_yaml(tmp_path, "test1", {})
    create_test_yaml(tmp_path, "test2", {})

    db = YamlDatabase(config_dir=tmp_path)
    values = list(db.values())

    assert len(values) == 2
    assert {} in values
    assert {} in values
