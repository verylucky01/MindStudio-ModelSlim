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
from typing import Dict, Any, Union
from pathlib import Path

from pydantic import BaseModel, Field, model_validator
from typing_extensions import Self, Literal
import torch.nn as nn

from msmodelslim.core.quant_service.interface import BaseQuantConfig
from msmodelslim.core.quant_service.modelslim_v1.quant_config import ModelslimV1QuantConfig, ModelslimV1ServiceConfig
from msmodelslim.utils.exception import SchemaValidateError
from msmodelslim.utils.exception_decorator import exception_handler
from .pipeline_interface import MultimodalPipelineInterface


class DumpConfig(BaseModel):
    capture_mode: Literal["args"] = Field(default="args")
    dump_data_dir: str = Field(default="")


# 多模态基础配置
class MultimodalSDConfig(BaseModel):
    dump_config: DumpConfig
    # 允许接收未定义的额外参数
    model_config = {
        "extra": "allow"
    }

    # 可选：将额外参数转换为字典属性，方便访问
    @property
    def extra_params(self) -> Dict[str, Any]:
        return self.model_extra or {}


class MultimodalSDServiceConfig(ModelslimV1ServiceConfig):
    # 支持直接传入字典作为配置，或使用 MultimodalSDConfig 实例
    multimodal_sd_config: Union[Dict[str, Any], MultimodalSDConfig] = Field(
        default_factory=lambda: MultimodalSDConfig().model_dump()
    )

    # 验证并转换配置格式
    @model_validator(mode="after")
    def normalize_config(self) -> Self:
        if isinstance(self.multimodal_sd_config, dict):
            # 将字典转换 MultimodalSDConfig 实例（会保留额外字段）
            self.multimodal_sd_config = MultimodalSDConfig(**self.multimodal_sd_config)
        return self


class MultimodalSDModelslimV1QuantConfig(ModelslimV1QuantConfig):
    """支持多模态的量化配置类"""
    spec: MultimodalSDServiceConfig  # 使用新的多模态配置

    @classmethod
    def from_base(cls, quant_config: BaseQuantConfig) -> Self:
        return cls(
            apiversion=quant_config.apiversion,
            spec=load_specific_config(quant_config.spec),
        )


@exception_handler(err_cls=Exception, ms_err_cls=SchemaValidateError,
                   keyword="validation error",
                   action="Please check the multimodal_sd_config parameter of the YAML file.")
def load_specific_config(yaml_spec: object) -> MultimodalSDServiceConfig:
    """Load specific configuration from YAML spec"""
    if isinstance(yaml_spec, MultimodalSDServiceConfig):
        return yaml_spec
    if not isinstance(yaml_spec, dict):
        raise SchemaValidateError("task spec must be dict")
    return MultimodalSDServiceConfig.model_validate(yaml_spec)


@dataclass
class MultiExpertQuantConfig:
    """多专家模型量化配置"""
    model_adapter: MultimodalPipelineInterface
    models: dict[str, nn.Module]
    calib_data: dict[str, Any]
    quant_config: MultimodalSDModelslimV1QuantConfig
    save_path: Path
    device: str = "npu"
