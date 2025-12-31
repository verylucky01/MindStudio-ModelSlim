# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
from abc import abstractmethod
from typing import Generator, Optional

from msmodelslim.core.practice import PracticeConfig
from msmodelslim.core.tune_strategy import ITuningStrategy
from msmodelslim.core.tune_strategy.interface import StrategyConfig, EvaluateResult
from msmodelslim.core.const import DeviceType
from msmodelslim.model import IModel
from .dataset_loader_infra import DatasetLoaderInfra


class BaseTuningStrategy(ITuningStrategy):
    def __init__(self, config: StrategyConfig, dataset_loader: DatasetLoaderInfra):
        self.config = config
        self.dataset_loader = dataset_loader

    @abstractmethod
    def generate_practice(self,
                          model: IModel,
                          device: DeviceType = DeviceType.NPU,
                          ) -> Generator[PracticeConfig, Optional[EvaluateResult], None]:
        ...
