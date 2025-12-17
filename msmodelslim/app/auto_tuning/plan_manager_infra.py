# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
from abc import ABC, abstractmethod

from pydantic import BaseModel, SerializeAsAny

from msmodelslim.app.tune_strategy import StrategyConfig
from .evaluation_service_infra import EvaluateServiceConfig


class TuningPlanConfig(BaseModel):
    strategy: SerializeAsAny[StrategyConfig]
    evaluation: SerializeAsAny[EvaluateServiceConfig]


class TuningPlanManagerInfra(ABC):
    @abstractmethod
    def get_plan_by_id(self, plan_id: str) -> TuningPlanConfig:
        ...
