# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
"""
插件工具函数：提供插件加载的共用逻辑
"""
import sys
import traceback
from importlib.metadata import entry_points
from typing import Type

from msmodelslim.utils.exception import ToDoError, UnsupportedError
from msmodelslim.utils.logging import get_logger


def get_entry_points(group_name: str):
    """获取 entry_points，兼容不同 Python 版本"""
    if sys.version_info >= (3, 10):
        return entry_points().select(group=group_name)
    return entry_points().get(group_name, [])


def load_plugin_class(
        entry_point_group: str,
        plugin_type: str,
        base_class: Type
) -> Type:
    """
    根据类型加载对应的插件类（每次都重新加载，不使用缓存）
    
    Args:
        entry_point_group: entry_point 组名
        plugin_type: 插件类型字符串
        base_class: 基类，用于验证加载的类是否为子类
        
    Returns:
        插件类
    """
    # 查找并加载插件
    for entry in get_entry_points(entry_point_group):
        if entry.name != plugin_type:
            continue

        try:
            plugin_class = entry.load()
            if not issubclass(plugin_class, base_class):
                error_msg = f"Plugin {plugin_type} is not a subclass of {base_class.__name__}"
                get_logger().error(
                    "[plugin_utils] Plugin %r from group %r is not a subclass of %r",
                    plugin_type,
                    entry_point_group,
                    base_class.__name__,
                )
                raise ToDoError(error_msg)

            get_logger().debug(
                "[plugin_utils] Load plugin %r from group %r success!",
                plugin_type,
                entry_point_group,
            )
            return plugin_class
        except Exception as e:
            error_msg = f"{str(e)}\n{traceback.format_exc()}"
            get_logger().error(
                "[plugin_utils] Failed to load plugin %r from group %r: %r",
                plugin_type,
                entry_point_group,
                e,
            )
            raise ToDoError(
                f"Plugin for type '{plugin_type}' in group '{entry_point_group}' failed to load:\n{error_msg}"
            ) from e

    raise UnsupportedError(
        f"No plugin found for type '{plugin_type}' in group '{entry_point_group}'.",
        action=f"Please install plugin before using."
    )
