#  Copyright (c) 2025-2025 Huawei Technologies Co., Ltd.
import sys
import traceback
from importlib.metadata import entry_points
from pathlib import Path
from typing import Optional, Any, List, Dict, Type

from msmodelslim.core.quant_service import BaseQuantService, DatasetLoaderInfra
from msmodelslim.core.quant_service.interface import BaseQuantConfig
from msmodelslim.core.const import DeviceType
from msmodelslim.utils.exception import EnvError
from msmodelslim.utils.logging import logger_setter, get_logger

_QUANT_SERVICE_PLUGINS: Dict[str, Type[BaseQuantService]] = {}
_FAILED_PLUGINS: Dict[str, str] = {}  # 记录加载失败的插件及其错误信息


@logger_setter(prefix='msmodelslim.core.quant_service.proxy')  # use 4-level path: msmodelslim.core.quant_service.proxy
class QuantServiceProxy(BaseQuantService):

    def __init__(self, dataset_loader: DatasetLoaderInfra, vlm_dataset_loader: DatasetLoaderInfra):
        super().__init__(dataset_loader)
        self.quant_service: Optional[BaseQuantService] = None
        self.vlm_dataset_loader = vlm_dataset_loader

    def quantize(
            self,
            quant_config: BaseQuantConfig,
            model_adapter: Any,
            save_path: Optional[Path] = None,
            device: DeviceType = DeviceType.NPU,
            device_indices: Optional[List[int]] = None,
    ) -> None:
        load_plugins()

        # Determine the appropriate dataset loader based on apiversion
        self._set_dataset_loader_for_service(quant_config.apiversion)

        self.quant_service = load_quant_service_cls(quant_config.apiversion)(self.dataset_loader)
        self.quant_service.quantize(
            quant_config=quant_config,
            model_adapter=model_adapter,
            save_path=save_path,
            device=device,
            device_indices=device_indices
        )
    
    def _set_dataset_loader_for_service(self, apiversion: str) -> DatasetLoaderInfra:
        """
        Set the appropriate dataset loader for a specific service.
        
        For services that require specialized dataset loaders, 
        set the appropriate loader instance.
        
        For other services, set the dataset_loader to the default FileDatasetLoader.
        
        Args:
            apiversion: The API version string (e.g., "multimodal_vlm_modelslim_v1")
        
        Returns:
            Dataset loader instance
        """
        # Map services to their specialized dataset loaders
        if apiversion == 'multimodal_vlm_modelslim_v1':
            self.dataset_loader = self.vlm_dataset_loader
        # Other services use the default FileDatasetLoader


def get_entry_points(group_name):
    if sys.version_info >= (3, 10):
        # Python 3.10+ 使用新API
        return entry_points().select(group=group_name)

    # Python 3.8-3.9 使用旧API
    return entry_points().get(group_name, [])


def load_plugins() -> Dict[str, Type[BaseQuantService]]:
    for entry in get_entry_points(group_name="msmodelslim.quant_service.plugins"):
        try:
            plugin_class = entry.load()
            if issubclass(plugin_class, BaseQuantService):
                _QUANT_SERVICE_PLUGINS[plugin_class.backend_name] = plugin_class
                get_logger().info(f"Load quant service plugin {entry.name} success!")
            else:
                error_msg = f"Plugin {entry.name} is not a subclass of BaseQuantService"
                _FAILED_PLUGINS[entry.name] = error_msg
                get_logger().warning(f"Failed to load plugin {entry.name}: {error_msg}")
        except Exception as e:
            error_msg = f"Exception: {str(e)}\nTraceback: {traceback.format_exc()}"
            _FAILED_PLUGINS[entry.name] = error_msg
            get_logger().warning(f"Failed to load plugin {entry.name}: {e}")
    return _QUANT_SERVICE_PLUGINS


def load_quant_service_cls(backend_name: str) -> Type[BaseQuantService]:
    # 首先检查请求的后端是否在失败列表中
    if backend_name in _FAILED_PLUGINS:
        raise EnvError(
            f"Quant service plugin for backend '{backend_name}' failed to load:\n{_FAILED_PLUGINS[backend_name]}")

    if backend_name not in _QUANT_SERVICE_PLUGINS:
        available_backends = list(_QUANT_SERVICE_PLUGINS.keys())
        raise EnvError(
            f"No quant service plugin found for backend '{backend_name}'. Available backends: {available_backends}")

    return _QUANT_SERVICE_PLUGINS[backend_name]
