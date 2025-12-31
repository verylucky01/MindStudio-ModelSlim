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
from dataclasses import field
from typing import List

from pydantic import Field, BaseModel

from msmodelslim.core.quant_service.interface import BaseQuantConfig


class Metadata(BaseModel):
    # ID of the quantization config, e.g., 'Qwen3-32B W8A8'
    config_id: str = 'Unknown'
    # score of the quantization config, used to sort the quantization configs
    score: float = 100.0
    # label of the quantization config, used to filter the quantization configs.
    # e.g., # {'w_bit': 8, 'a_bit': 8, 'is_sparse': True, 'kv_cache': True}
    label: dict = Field(default_factory=dict)
    # verified model types, e.g., ['LLaMa3.1-70B', 'Qwen2.5-72B']
    verified_model_types: List[str] = field(default_factory=list)


class PracticeConfig(BaseQuantConfig):
    metadata: Metadata = Field(default_factory=Metadata) # metadata of the quantization config

    def extract_quant_config(self) -> BaseQuantConfig:
        return self
