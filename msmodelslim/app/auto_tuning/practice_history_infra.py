# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
from abc import ABC, abstractmethod

from pydantic import BaseModel

from msmodelslim.app.practice import PracticeConfig
from msmodelslim.app.tune_strategy.interface import EvaluateResult


class TuningHistory(BaseModel):
    practice: PracticeConfig
    evaluation: EvaluateResult


class TuningHistoryManagerInfra(ABC):
    @abstractmethod
    def append_history(self, database: str, history: TuningHistory) -> None:
        ...
