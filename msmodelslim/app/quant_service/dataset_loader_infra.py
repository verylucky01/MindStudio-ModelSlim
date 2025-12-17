# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
from abc import ABC, abstractmethod
from typing import List


class DatasetLoaderInfra(ABC):
    @abstractmethod
    def get_dataset_by_name(self, dataset_id: str) -> List[str]:
        """Get configuration by ID"""
        raise NotImplementedError
