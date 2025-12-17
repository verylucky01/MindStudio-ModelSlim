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
from dataclasses import dataclass

from pydantic import BaseModel, Field, ConfigDict
from typing_extensions import Self

from msmodelslim.core.const import RunnerType
from msmodelslim.quant.processor.base import AutoProcessorConfigList
from .save.saver import AutoSaverConfigList
from ..interface import BaseQuantConfig


class ModelslimV1ServiceConfig(BaseModel):
    runner: RunnerType = RunnerType.AUTO
    process: AutoProcessorConfigList = Field(default_factory=list)
    save: AutoSaverConfigList = Field(default_factory=list)
    dataset: str = Field(default='mix_calib.jsonl')

    model_config = ConfigDict(use_enum_values=True)


@dataclass
class ModelslimV1QuantConfig(BaseQuantConfig):
    spec: ModelslimV1ServiceConfig  # quantization config specification

    @classmethod
    def from_base(cls, quant_config: BaseQuantConfig) -> Self:
        return cls(
            apiversion=quant_config.apiversion,
            spec=load_specific_config(quant_config.spec),
        )


def load_specific_config(yaml_spec: object) -> ModelslimV1ServiceConfig:
    """Load specific configuration from YAML spec"""
    if not isinstance(yaml_spec, dict):
        raise ValueError("task spec must be dict")
    return ModelslimV1ServiceConfig.model_validate(yaml_spec)
