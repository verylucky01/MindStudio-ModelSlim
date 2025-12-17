#  Copyright (c) 2025-2025 Huawei Technologies Co., Ltd.
from abc import abstractmethod
from pathlib import Path
from typing import Optional, List

from msmodelslim.app.quant_service.dataset_loader_infra import DatasetLoaderInfra
from msmodelslim.core.const import DeviceType
from .interface import IQuantService, BaseQuantConfig
from ...model import IModel


class BaseQuantService(IQuantService):
    backend_name: str = "Unknown"

    def __init__(self, dataset_loader: DatasetLoaderInfra):
        self.dataset_loader = dataset_loader

    @abstractmethod
    def quantize(self,
                 quant_config: BaseQuantConfig,
                 model_adapter: IModel,
                 save_path: Optional[Path] = None,
                 device: DeviceType = DeviceType.NPU,
                 device_indices: Optional[List[int]] = None
                 ) -> None:
        ...
