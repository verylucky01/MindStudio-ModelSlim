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

from pathlib import Path

from .interface_hub import IModel


class BaseModelAdapter(IModel):

    def __init__(self, model_type: str, model_path: Path, trust_remote_code: bool = False):
        self._model_type = model_type
        self._model_path = model_path
        self._trust_remote_code = trust_remote_code

    @property
    def model_type(self) -> str:
        return self._model_type

    @property
    def model_path(self) -> Path:
        return self._model_path

    @property
    def trust_remote_code(self) -> bool:
        return self._trust_remote_code

    @model_type.setter
    def model_type(self, model_type: str):
        self._model_type = model_type

    @model_path.setter
    def model_path(self, model_path: Path):
        self._model_path = model_path

    @trust_remote_code.setter
    def trust_remote_code(self, trust_remote_code: bool):
        self._trust_remote_code = trust_remote_code
