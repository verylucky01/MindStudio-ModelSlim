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
from abc import ABC, abstractmethod
from pathlib import Path


class IModel(ABC):
    """
    Base Interface for the model.
    Just show the basic properties of the model.
    """

    @property
    @abstractmethod
    def model_type(self) -> str:
        ...

    @property
    @abstractmethod
    def model_path(self) -> Path:
        ...

    @property
    @abstractmethod
    def trust_remote_code(self) -> bool:
        ...


class IModelFactory(ABC):

    @abstractmethod
    def create(self,
               model_type: str,
               model_path: Path,
               trust_remote_code: bool = False,
               ) -> IModel:
        ...
