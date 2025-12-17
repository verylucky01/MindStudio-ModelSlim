# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
"""
类型化工厂类：根据 config 的 TypeField 动态加载对象类并用 config 初始化

使用示例:
    from msmodelslim.utils.plugin.typed_factory import TypedFactory
    from msmodelslim.app.quant_service.interface import QuantServiceConfig
    from msmodelslim.app.quant_service import IQuantService
    
    # 创建工厂实例（使用泛型指定返回类型）
    factory = TypedFactory[IQuantService](
        entry_point_group="msmodelslim.quant_service.plugins",
        config_base_class=QuantServiceConfig
    )
    
    # 使用 config 创建对象，返回类型为 IQuantService
    service = factory.create(config, **extra_kwargs)
"""
from typing import Type, TypeVar, Generic

from pydantic import BaseModel

from msmodelslim.utils.exception import ToDoError, UnsupportedError
from msmodelslim.utils.logging import get_logger
from msmodelslim.utils.plugin.plugin_utils import load_plugin_class
from msmodelslim.utils.plugin.typed_config import TypedConfig

T = TypeVar('T', bound=object)


class TypedFactory(Generic[T]):
    """
    类型化工厂类：根据 config 的 TypeField 动态加载对象类并用 config 初始化
    
    使用示例:
        from msmodelslim.utils.plugin.typed_factory import TypedFactory
        from msmodelslim.app.quant_service.interface import QuantServiceConfig
        from msmodelslim.app.quant_service import IQuantService
        
        # 创建工厂实例（使用泛型指定返回类型）
        factory = TypedFactory[IQuantService](
            entry_point_group="msmodelslim.quant_service.plugins",
            config_base_class=QuantServiceConfig
        )
        
        # 使用 config 创建对象，返回类型为 IQuantService
        service = factory.create(config, **extra_kwargs)
    """

    def __init__(
            self,
            entry_point_group: str,
            config_base_class: Type[BaseModel]
    ):
        """
        初始化类型化工厂
        
        Args:
            entry_point_group: entry_point 组名，例如 "msmodelslim.quant_service.plugins"
            config_base_class: 配置基类，用于检测 TypeField
        """
        self.entry_point_group = entry_point_group
        self.config_base_class = config_base_class
        self.type_field = TypedConfig.detect_type_field(config_base_class)

    def create(self, config: BaseModel, *args, **kwargs) -> T:
        """
        根据 config 创建对象实例
        
        Args:
            config: 配置对象，必须包含 TypeField 字段
            *args: 传递给对象类 __init__ 的额外位置参数
            **kwargs: 传递给对象类 __init__ 的额外关键字参数
        
        Returns:
            动态加载的类实例
        """
        # 验证 config 类型
        if not isinstance(config, self.config_base_class):
            raise UnsupportedError(
                f"Config must be an instance of {self.config_base_class.__name__}, "
                f"got {type(config).__name__}"
            )

        # 从 config 中获取类型字段值
        plugin_type = getattr(config, self.type_field, None)
        if not plugin_type:
            raise ToDoError(f"Attr {self.type_field} is required in the configuration")

        # 动态加载类（每次都重新加载，不使用缓存；不进行类型校验，传入 object 作为基类）
        plugin_class = load_plugin_class(self.entry_point_group, plugin_type, object)

        # 使用 config 和额外参数初始化对象
        # 首先尝试将 config 作为第一个参数传递
        try:
            instance = plugin_class(config, *args, **kwargs)
        except TypeError:
            # 如果直接传递 config 失败，尝试将 config 作为关键字参数传递
            try:
                instance = plugin_class(*args, config=config, **kwargs)
            except TypeError as e:
                # 如果还是失败，尝试将 config 的属性展开传递
                if isinstance(config, BaseModel):
                    config_dict = config.model_dump()
                    # 合并 kwargs，kwargs 中的值优先
                    merged_kwargs = {**config_dict, **kwargs}
                    instance = plugin_class(*args, **merged_kwargs)
                else:
                    raise ToDoError(
                        f"Failed to instantiate {plugin_class.__name__} with config. "
                        f"Error: {str(e)}. "
                        f"Please check the __init__ signature of {plugin_class.__name__}"
                    ) from e

        get_logger().debug(
            "[typed_factory] Successfully created %r instance from config with %r=%r",
            plugin_class.__name__,
            self.type_field,
            plugin_type,
        )

        return instance
