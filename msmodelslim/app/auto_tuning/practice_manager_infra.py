# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
from abc import ABC, abstractmethod

from msmodelslim.app.practice import PracticeConfig


class PracticeManagerInfra(ABC):
    @abstractmethod
    def save_practice(self, model_pedigree: str, practice: PracticeConfig) -> None:
        ...

    @abstractmethod
    def is_saving_supported(self) -> bool:
        ...
