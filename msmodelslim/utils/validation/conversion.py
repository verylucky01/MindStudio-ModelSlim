# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
from pathlib import Path

from msmodelslim.utils.exception import SchemaValidateError, UnsupportedError
from msmodelslim.utils.security import get_valid_read_path, get_write_directory


def convert_to_bool(obj: object) -> bool:
    if obj == "True":
        return True
    elif obj == "False":
        return False
    raise SchemaValidateError(f"{obj} is not True or False",
                              action=f"Please ensure the input is literally True or False")


def convert_to_readable_dir(obj: object) -> Path:
    if isinstance(obj, str):
        obj = get_valid_read_path(obj, is_dir=True)
        return Path(obj)
    raise UnsupportedError(f"Unsupported type converted to readable dir: {type(obj)}",
                           action=f"Please ensure the input is a string")


def convert_to_writable_dir(obj: object) -> Path:
    if isinstance(obj, str):
        obj = get_write_directory(obj, write_mode=0o750)
        return Path(obj)
    raise UnsupportedError(f"Unsupported type converted to writable dir: {type(obj)}",
                           action=f"Please ensure the input is a string")


def convert_to_readable_file(obj: object) -> Path:
    if isinstance(obj, str):
        obj = get_valid_read_path(obj, is_dir=False)
        return Path(obj)
    raise UnsupportedError(f"Unsupported type converted to readable file: {type(obj)}",
                           action=f"Please ensure the input is a string")
