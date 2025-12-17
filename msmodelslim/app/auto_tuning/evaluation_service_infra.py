# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Optional

from pydantic import BaseModel

from msmodelslim.app.tune_strategy.interface import EvaluateResult
from msmodelslim.core.const import DeviceType
from msmodelslim.utils.plugin import TypedConfig

EVALUATE_CONFIG_PLUGIN_PATH = "msmodelslim.evaluate_config.plugins"


class EvaluateContext(BaseModel):
    evaluate_id: str
    device: DeviceType = DeviceType.NPU
    device_indices: Optional[List[int]] = None
    working_dir: Path


@TypedConfig.plugin_entry(entry_point_group=EVALUATE_CONFIG_PLUGIN_PATH)
class EvaluateServiceConfig(TypedConfig):
    type: TypedConfig.TypeField


class EvaluateServiceInfra(ABC):
    @abstractmethod
    def evaluate(self,
                 context: EvaluateContext,
                 evaluate_config: EvaluateServiceConfig,
                 model_path: Path,
                 ) -> EvaluateResult:
        ...
