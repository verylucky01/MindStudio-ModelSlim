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

import importlib
from dataclasses import dataclass
from types import ModuleType
from typing import Optional, Dict

from packaging import version

from msmodelslim.utils.exception import VersionError
from msmodelslim.utils.logging import get_logger


@dataclass
class FuncMark:
    min_version: str
    max_version: str


class KIAManager:
    MAX_VERSION: str = "9999.9999.9999"
    KIA_MODULE: Optional[ModuleType] = None
    INITIALIZED: bool = False
    MARKED_FUNC: Dict[str, FuncMark] = {}

    @classmethod
    def init_module(cls):
        if cls.INITIALIZED:
            return

        if cls.KIA_MODULE is not None:
            return

        try:
            cls.KIA_MODULE = importlib.import_module("..impl.kia_main", __name__)
        except ImportError as e:
            get_logger().warning(
                f"[Core] Load KIA module failed, please check if the kia_impl package is installed, "
                f"exception info : {e}")

        cls.INITIALIZED = True

    @classmethod
    def get_module(cls) -> ModuleType:
        KIAManager.init_module()
        return cls.KIA_MODULE

    @classmethod
    def get_version(cls) -> str:
        KIAManager.init_module()
        return cls.KIA_MODULE.__version__ if cls.KIA_MODULE is not None else "0.0.0"

    @classmethod
    def check_version(cls):
        kia_version = version.parse(cls.get_version())
        for func_name, mark in cls.MARKED_FUNC.items():
            if kia_version < version.parse(mark.min_version) or kia_version > version.parse(mark.max_version):
                get_logger().warning(
                    f"Function {func_name} requires KIA version {mark.min_version}-{mark.max_version}, "
                    f"but the current KIA version is {cls.get_version()}")

    @classmethod
    def mark_require_version(cls, min_version: str, max_version: str = MAX_VERSION):

        """
        检查KIA版本是否满足要求。
        
        通过装饰器，可以在函数定义、函数运行两个阶段检查版本是否符合要求。
        
        在定义函数阶段，如果版本不满足要求，会抛出ImportError异常。

        在函数运行阶段，如果版本不满足要求，会抛出RuntimeError异常。
        
        Args:
            min_version: 最小版本
            max_version: 最大版本

        Raises:
            ImportError: 函数定义阶段，版本不满足时抛出异常
            RuntimeError: 函数运行阶段，版本不满足时抛出异常
        """

        def decorator(func):
            cls.MARKED_FUNC[func.__name__] = FuncMark(min_version, max_version)

            def wrapped_func(*args, **kwargs):
                kia_version = version.parse(cls.get_version())
                func_name = func.__name__

                # Runtime Check

                if kia_version < version.parse(min_version) or kia_version > version.parse(max_version):
                    raise VersionError(
                        f"Trying to invoke function {func_name}, "
                        f"which requires KIA version {min_version}-{max_version}, "
                        f"but the current KIA version is {cls.get_version()}",
                        action=f"Please upgrade CANN to {min_version}-{max_version}"
                    )

                return func(*args, **kwargs)

            return wrapped_func

        return decorator
