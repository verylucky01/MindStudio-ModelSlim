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

import sys

from pydantic import BaseModel as PydanticBaseModel, ValidationError

from msmodelslim.utils.exception import SchemaValidateError


def patch_pydantic():
    """
    通过patch的方式替换pydantic的BaseModel，将ValidationError转换为SchemaValidateError
    """

    # 保存原始的BaseModel
    original_base_model = PydanticBaseModel

    class PatchedBaseModel(original_base_model):
        """
        自定义BaseModel，将pydantic的ValidationError转换为项目的SchemaValidateError
        """

        def __init__(self, *args, **kwargs):
            try:
                super().__init__(*args, **kwargs)
            except ValidationError as e:
                raise SchemaValidateError(str(e),
                                          action="Please fix the data schema error and retry."
                                          ) from e

        @classmethod
        def model_validate(cls, *args, **kwargs):
            try:
                return super().model_validate(*args, **kwargs)
            except ValidationError as e:
                raise SchemaValidateError(str(e),
                                          action="Please fix the data schema error and retry."
                                          ) from e

        @classmethod
        def model_validate_json(cls, *args, **kwargs):
            try:
                return super().model_validate_json(*args, **kwargs)
            except ValidationError as e:
                raise SchemaValidateError(str(e),
                                          action="Please fix the data schema error and retry."
                                          ) from e

    # 全局替换pydantic的BaseModel
    import pydantic
    pydantic.BaseModel = PatchedBaseModel

    # 替换sys.modules中的BaseModel，确保所有import都能获取到patched版本
    if 'pydantic' in sys.modules:
        sys.modules['pydantic'].BaseModel = PatchedBaseModel

    # 为了确保所有可能的导入路径都被覆盖
    for module_name in list(sys.modules.keys()):
        if module_name.startswith('pydantic'):
            module = sys.modules[module_name]
            if hasattr(module, 'BaseModel'):
                module.BaseModel = PatchedBaseModel
