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
msmodelslim.utils.security.path 模块的单元测试
"""
import os
import stat
import json
import sys
import yaml
import pytest
from msmodelslim.utils.exception import SecurityError, SchemaValidateError

TEST_DIR = "/tmp/a_test_path_for_testing_automl_common/"
TEST_READ_FILE_NAME = TEST_DIR + "testfile.testfile"
USER_NOT_PERMITTED_READ_FILE = TEST_DIR + "testfile_not_readable.testfile"
OTHERS_READABLE_READ_FILE = TEST_DIR + "testfile_others_readable.testfile"
OTHERS_WRITABLE_READ_FILE = TEST_DIR + "testfile_others_writable.testfile"
USER_NOT_PERMITTED_WRITE_FILE = TEST_DIR + "testfile_not_writable/foo"
JSON_FILE = TEST_DIR + "testfile.json"
YAML_FILE = TEST_DIR + "testfile.yaml"
TEST_FILE = TEST_DIR + "testfile.test"
ORI_DATA = {
    "a_long_key_name": 1,
    12: "b",
    3.14: "",
    "c": {"d": 3, "e": 4},
    True: "true",
    False: "false",
    None: "null",
}
OVER_WRITE_DATA = {"hello": "world"}


def setup_module():
    os.makedirs(TEST_DIR, mode=int("700", 8), exist_ok=True)

    default_mode = stat.S_IWUSR | stat.S_IRUSR  # 600
    with os.fdopen(
        os.open(
            TEST_READ_FILE_NAME,
            os.O_WRONLY | os.O_CREAT | os.O_TRUNC,
            mode=default_mode,
        ),
        "w",
    ) as temp_file:
        temp_file.write("a_test_file_name_for_testing_automl_common")

    with os.fdopen(os.open(USER_NOT_PERMITTED_READ_FILE, os.O_CREAT, mode=000), "w"):
        pass

    with os.fdopen(
        os.open(OTHERS_READABLE_READ_FILE, os.O_CREAT, mode=default_mode), "w"
    ):
        pass
    os.chmod(OTHERS_READABLE_READ_FILE, int("755", 8))

    with os.fdopen(
        os.open(OTHERS_WRITABLE_READ_FILE, os.O_CREAT, mode=default_mode), "w"
    ):
        pass
    os.chmod(OTHERS_WRITABLE_READ_FILE, int("666", 8))

    dir_name = os.path.dirname(USER_NOT_PERMITTED_WRITE_FILE)
    os.makedirs(dir_name, mode=int("500", 8), exist_ok=True)

    with os.fdopen(
        os.open(JSON_FILE, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, mode=default_mode),
        "w",
    ) as json_file:
        json.dump(ORI_DATA, json_file)

    with os.fdopen(
        os.open(YAML_FILE, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, mode=default_mode),
        "w",
    ) as yaml_file:
        yaml.dump(ORI_DATA, yaml_file)

    module_name = "msmodelslim.utils.security.path"
    if module_name in sys.modules:
        del sys.modules[module_name]


def teardown_module():
    os.remove(TEST_READ_FILE_NAME)
    os.chmod(USER_NOT_PERMITTED_READ_FILE, int("600", 8))
    os.remove(USER_NOT_PERMITTED_READ_FILE)
    os.remove(OTHERS_READABLE_READ_FILE)
    os.remove(OTHERS_WRITABLE_READ_FILE)

    dir_name = os.path.dirname(USER_NOT_PERMITTED_WRITE_FILE)
    os.chmod(dir_name, int("700", 8))
    os.removedirs(dir_name)

    os.remove(JSON_FILE)
    os.remove(YAML_FILE)
    if os.path.exists(TEST_FILE):
        os.remove(TEST_FILE)

    os.removedirs(TEST_DIR)


def test_get_valid_path_given_valid_when_any_then_pass():
    """测试 get_valid_path：当路径合法时，函数应正常通过（不抛异常）"""
    from msmodelslim.utils.security.path import get_valid_path

    get_valid_path("../anypath")
    get_valid_path("../anypath/a")


def test_get_valid_path_given_invalid_when_any_then_value_error():
    """测试 get_valid_path：当路径包含非法字符时，应抛出 SecurityError"""
    from msmodelslim.utils.security.path import get_valid_path

    with pytest.raises(SecurityError):
        get_valid_path("../anypath*a")
    with pytest.raises(SecurityError):
        get_valid_path("../anypath/\\a")
    with pytest.raises(SecurityError):
        get_valid_path("../anypath/!a")


def test_get_valid_read_path_given_valid_when_any_then_pass():
    """测试 get_valid_read_path：合法文件路径应成功通过安全性检查"""
    from msmodelslim.utils.security.path import get_valid_read_path

    get_valid_read_path(TEST_READ_FILE_NAME)
    get_valid_read_path(TEST_READ_FILE_NAME, extensions=".testfile")
    get_valid_read_path(OTHERS_READABLE_READ_FILE)
    get_valid_read_path(OTHERS_WRITABLE_READ_FILE, check_user_stat=False)


def test_get_valid_read_path_given_invalid_when_any_then_value_error():
    """测试 get_valid_read_path：非法或不符合要求的文件应抛 SecurityError"""
    from msmodelslim.utils.security.path import get_valid_read_path

    with pytest.raises(SecurityError):
        get_valid_read_path(
            "./not_exist"
        )  # SecurityError: The file ... doesn't exist or not a file.
    with pytest.raises(SecurityError):
        # SecurityError: The filename ... doesn't endswith ".json"
        get_valid_read_path(TEST_READ_FILE_NAME, extensions=".json")
    with pytest.raises(SecurityError):
        # SecurityError: The file ... exceeds size limitation of 1.
        get_valid_read_path(TEST_READ_FILE_NAME, size_max=1)
    with pytest.raises(SecurityError):
        # SecurityError: Current user doesn't have read permission to the file ....
        get_valid_read_path(USER_NOT_PERMITTED_READ_FILE)
    with pytest.raises(SecurityError):
        # SecurityError: The file ... has others writable permission.
        get_valid_read_path(OTHERS_WRITABLE_READ_FILE)


def test_check_write_directory_given_valid_when_any_then_pass():
    """测试 check_write_directory：合法目录应通过检查"""
    from msmodelslim.utils.security.path import check_write_directory

    check_write_directory(TEST_DIR)


def test_check_write_directory_given_invalid_when_any_then_error():
    """测试 check_write_directory：目录不存在时应抛 SecurityError"""
    from msmodelslim.utils.security.path import check_write_directory

    with pytest.raises(SecurityError):
        # SecurityError: The file writen directory ... doesn't exist.
        check_write_directory("not_exists")


def test_get_write_directory_given_valid_when_any_then_pass():
    """测试 get_write_directory：合法目录应被正确返回"""
    from msmodelslim.utils.security.path import get_write_directory

    get_write_directory(TEST_DIR)


def test_get_write_directory_given_invalid_when_any_then_error():
    """测试 get_write_directory：目录不存在时应抛异常"""
    from msmodelslim.utils.security.path import get_write_directory

    get_write_directory("not_exists_")


def test_get_valid_write_path_given_valid_when_any_then_pass():
    """测试 get_valid_write_path：合法文件路径应允许写入（可能覆盖旧文件）"""
    from msmodelslim.utils.security.path import get_valid_write_path

    get_valid_write_path(TEST_READ_FILE_NAME, extensions=".testfile")


def test_get_valid_write_path_when_directory_not_exists():
    """测试 get_valid_write_path：当目录不存在时应抛 SecurityError"""
    from msmodelslim.utils.security.path import get_valid_write_path

    with pytest.raises(SecurityError):
        # SecurityError: The file writen directory ... doesn't exist.
        get_valid_write_path("not_exists/README.md", extensions=".md")


@pytest.mark.skipif(
    os.geteuid() == 0,  # 直接判断：如果是 root 用户（UID=0）
    reason="root 用户跳过此用例",
)
def test_get_valid_write_path_when_no_write_permission():
    from msmodelslim.utils.security.path import get_valid_write_path

    """测试 get_valid_write_path：当前用户对目录无写权限时应抛 SecurityError"""
    with pytest.raises(SecurityError):
        # SecurityError: Current user doesn't have writen permission to the file writen directory ....
        get_valid_write_path(USER_NOT_PERMITTED_WRITE_FILE)


def test_yaml_safe_load_given_valid_when_any_then_pass():
    """测试 yaml_safe_load：合法 YAML 文件应成功加载"""
    from msmodelslim.utils.security.path import yaml_safe_load

    yaml_safe_load(YAML_FILE)


def test_yaml_safe_load_given_invalid_when_any_then_value_error():
    """测试 yaml_safe_load：非法或不符合格式的 YAML 文件应抛异常"""
    from msmodelslim.utils.security.path import yaml_safe_load

    with pytest.raises(SecurityError):
        # SecurityError: The filename ... doesn't endswith "['.yml', '.yaml']".
        yaml_safe_load(TEST_READ_FILE_NAME)
    with pytest.raises(SchemaValidateError):
        # SecurityError: Length of ... exceeds key limitation of 2.
        yaml_safe_load(YAML_FILE, key_max_len=2)


def test_json_safe_load_given_valid_when_any_then_pass():
    """测试 json_safe_load：合法 JSON 文件应成功加载"""
    from msmodelslim.utils.security.path import json_safe_load

    json_safe_load(JSON_FILE)


def test_json_safe_load_given_invalid_when_any_then_value_error():
    """测试 json_safe_load：非法 JSON 文件应抛 SecurityError"""
    from msmodelslim.utils.security.path import json_safe_load

    with pytest.raises(SecurityError):
        # SecurityError: The filename ... doesn't endswith ".json"
        json_safe_load(YAML_FILE)
    with pytest.raises(SecurityError):
        # SecurityError: The file ... exceeds size limitation of 1.
        json_safe_load(JSON_FILE, size_max=1)


def test_file_safe_write_given_valid_when_any_then_pass():
    """测试 file_safe_write：合法字符串内容应能安全写入文件"""
    from msmodelslim.utils.security.path import file_safe_write

    file_safe_write("hello world", TEST_FILE, ".test")


def test_file_safe_write_given_invalid_when_any_then_type_error():
    """测试 file_safe_write：写入非字符串对象应抛 SecurityError"""
    from msmodelslim.utils.security.path import file_safe_write

    with pytest.raises(SecurityError):
        # TypeError: obj must be str
        file_safe_write(ORI_DATA, TEST_FILE, ".test")


def test_yaml_safe_dump_given_valid_when_over_write_then_pass():
    """测试 yaml_safe_dump：可多次覆盖写入 YAML 文件并验证写入内容"""
    from msmodelslim.utils.security.path import yaml_safe_dump, yaml_safe_load

    yaml_safe_dump(ORI_DATA, YAML_FILE)
    yaml_safe_dump(OVER_WRITE_DATA, YAML_FILE)
    cur_dict = yaml_safe_load(YAML_FILE)
    assert cur_dict == OVER_WRITE_DATA


def test_json_safe_dump_given_valid_when_over_write_then_pass():
    """测试 json_safe_dump：JSON 文件多次写入覆盖应保持一致"""
    from msmodelslim.utils.security.path import json_safe_dump, json_safe_load

    json_safe_dump(ORI_DATA, JSON_FILE, indent=4)
    json_safe_dump(OVER_WRITE_DATA, JSON_FILE)
    cur_dict = json_safe_load(JSON_FILE)
    assert cur_dict == OVER_WRITE_DATA


def test_safe_copy_file_given_valid_when_over_write_then_pass():
    """测试 safe_copy_file：文件可安全复制并删除"""
    from msmodelslim.utils.security.path import (
        safe_copy_file,
        safe_delete_path_if_exists,
    )

    dest_path = TEST_READ_FILE_NAME + "_copy"
    safe_copy_file(TEST_READ_FILE_NAME, dest_path)
    safe_delete_path_if_exists(dest_path)


def test_set_file_stat_when_any_file_then_pass():
    """测试 set_file_stat：修改文件权限为安全模式后能正常执行"""
    from msmodelslim.utils.security.path import (
        safe_copy_file,
        set_file_stat,
        safe_delete_path_if_exists,
    )

    dest_path = TEST_READ_FILE_NAME + "_copy"
    safe_copy_file(TEST_READ_FILE_NAME, dest_path)
    set_file_stat(dest_path)
    safe_delete_path_if_exists(dest_path)


def test_safe_write_umask_given_valid_when_any_then_pass():
    """测试 SafeWriteUmask：确保创建文件时权限安全（无组/其他用户写权限）"""
    from msmodelslim.utils.security.path import (
        SafeWriteUmask,
        safe_delete_path_if_exists,
    )

    safe_delete_path_if_exists(TEST_FILE)
    default_flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC
    fake_mode = (
        stat.S_IWUSR
        | stat.S_IRUSR
        | stat.S_IWGRP
        | stat.S_IRGRP
        | stat.S_IWOTH
        | stat.S_IROTH
    )  # 666
    with SafeWriteUmask(), os.fdopen(
        os.open(TEST_FILE, default_flags, mode=fake_mode), "w"
    ) as write_file:
        write_file.write("")
    assert (
        os.stat(TEST_FILE).st_mode
        & (stat.S_IWGRP | stat.S_IWOTH | stat.S_IROTH | stat.S_IXOTH)
        == 0
    )
