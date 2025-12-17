# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
from abc import ABC, abstractmethod
from typing import Generator, List, Optional

from pydantic import BaseModel, Field

from msmodelslim.app.practice.interface import PracticeConfig
from msmodelslim.core.const import DeviceType
from msmodelslim.model import IModel
from msmodelslim.utils.plugin import TypedConfig

TUNING_STRATEGY_CONFIG_PLUGIN_PATH = "msmodelslim.strategy_config.plugins"


class EvaluateAccuracy(BaseModel):
    dataset: str
    accuracy: float


class AccuracyExpectation(BaseModel):
    dataset: str
    target: float
    tolerance: float


class EvaluateResult(BaseModel):
    accuracies: List[EvaluateAccuracy] = Field(default_factory=list)
    expectations: List[AccuracyExpectation] = Field(default_factory=list)
    is_satisfied: bool


@TypedConfig.plugin_entry(entry_point_group=TUNING_STRATEGY_CONFIG_PLUGIN_PATH)
class StrategyConfig(TypedConfig):
    type: TypedConfig.TypeField


class ITuningStrategy(ABC):
    @abstractmethod
    def generate_practice(self,
                          model: IModel,
                          device: DeviceType = DeviceType.NPU,
                          ) -> Generator[PracticeConfig, Optional[EvaluateResult], None]:
        ...


class ITuningStrategyFactory(ABC):
    @abstractmethod
    def create_strategy(self, strategy_config: StrategyConfig) -> ITuningStrategy:
        ...
