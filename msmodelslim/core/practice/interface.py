#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""
-------------------------------------------------------------------------
This file is part of the MindStudio project.
Copyright (c) 2025 Huawei Technologies Co.,Ltd.

MindStudio is licensed under Mulan PSL v2.
You can use this software according to the terms and conditions of the Mulan PSL v2.
You may obtain a copy of Mulan PSL v2 at:

         http://license.coscl.org.cn/MulanPSL2

THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
See the Mulan PSL v2 for more details.
-------------------------------------------------------------------------
"""
from dataclasses import field
from typing import Dict, List, Optional

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
    # verified_tags: Dict[model_type, List[List[tags]]]
    # key: model_type; value: list of scenarios, each scenario is a list of tags (e.g. ["MindIE","Atlas_A2_Inference"], ["vLLM-Ascend","Atlas_A3_Inference"])
    verified_tags: Dict[str, List[List[str]]] = Field(default_factory=dict)


class PracticeConfig(BaseQuantConfig):
    metadata: Metadata = Field(default_factory=Metadata) # metadata of the quantization config

    def extract_quant_config(self) -> BaseQuantConfig:
        return self

    def matches_scenario_tags(self, model_type: str, scenario_tags: Optional[List[str]]) -> bool:
        """
        Return True if config's verified_tags has at least one scenario (for model_type)
        that contains ALL effective tags.
        """
        model_scenario = getattr(self.metadata, 'verified_tags', None) or {}
        scenarios = model_scenario.get(model_type, [])
        if not scenarios:
            return False
        if not scenario_tags:
            return True
        user_lower = [t.lower() for t in scenario_tags]
        for scenario_tags_list in scenarios:
            scenario_lower = [str(t).lower() for t in scenario_tags_list]
            if all(ut in scenario_lower for ut in user_lower):
                return True
        return False