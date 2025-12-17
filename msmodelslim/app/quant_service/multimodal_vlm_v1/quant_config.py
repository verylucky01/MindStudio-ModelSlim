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
from pydantic import BaseModel, Field, field_validator
from typing_extensions import Self

from msmodelslim.app.quant_service.modelslim_v1.quant_config import ModelslimV1QuantConfig, ModelslimV1ServiceConfig
from msmodelslim.utils.exception import SchemaValidateError
from msmodelslim.utils.exception_decorator import exception_handler
from msmodelslim.utils.validation.value import non_empty_string
from ..interface import BaseQuantConfig


class MultimodalVLMServiceConfig(ModelslimV1ServiceConfig):
    default_text: str = Field(
        default="Describe this image in detail.",
        description="Default prompt used for image-only calibration data when text is not provided."
    )

    @field_validator("default_text")
    @classmethod
    def validate_default_text(cls, v: str) -> str:
        return non_empty_string(v, "default_text")


@dataclass
class MultimodalVLMModelslimV1QuantConfig(ModelslimV1QuantConfig):
    """
    Quantization configuration for Multimodal VLM V1 service.
    
    Compatible with NaiveQuantizationApplication and best practice system.
    """
    spec: MultimodalVLMServiceConfig

    @classmethod
    def from_base(cls, quant_config: BaseQuantConfig) -> Self:
        """Convert from base config"""
        return cls(
            apiversion=quant_config.apiversion,
            spec=load_specific_config(quant_config.spec),
        )


@exception_handler(err_cls=Exception, ms_err_cls=SchemaValidateError,
                   keyword="validation error",
                   action="Please check the spec parameters of the YAML file.")
def load_specific_config(yaml_spec: object) -> MultimodalVLMServiceConfig:
    """Load specific configuration from YAML spec"""
    if not isinstance(yaml_spec, dict):
        raise ValueError("task spec must be dict")
    return MultimodalVLMServiceConfig.model_validate(yaml_spec)