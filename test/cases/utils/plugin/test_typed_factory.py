#  -*- coding: utf-8 -*-
#  Copyright (c) 2025-2025 Huawei Technologies Co., Ltd.
#  #
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#  #
#  http://www.apache.org/licenses/LICENSE-2.0
#  #
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
"""
msmodelslim.utils.plugin.typed_factory 模块的单元测试（unittest TestCase + pytest 辅助）
"""

import unittest
from unittest.mock import patch

from pydantic import BaseModel

from msmodelslim.utils.exception import ToDoError, UnsupportedError
from msmodelslim.utils.plugin import TypedConfig, TypedFactory


class TestTypedFactory(unittest.TestCase):
    """测试 TypedFactory 工厂类（一一对应 TypedFactory 代码类）"""

    class MyConfig(BaseModel):
        kind: TypedConfig.TypeField
        value: int = 0

    def test_create_raise_unsupported_error_when_config_type_mismatch(self):
        """当 config 不是指定基类实例时，应抛出 UnsupportedError"""

        factory = TypedFactory[object](
            entry_point_group="test.group",
            config_base_class=TestTypedFactory.MyConfig,
        )

        with self.assertRaises(UnsupportedError) as cm:
            factory.create(BaseModel())  # type: ignore[arg-type]

        msg = str(cm.exception)
        self.assertIn("Config must be an instance of MyConfig", msg)

    def test_create_raise_todo_error_when_missing_type_field_value(self):
        """当 config 中类型字段值为空时，应抛出 ToDoError"""

        factory = TypedFactory[object](
            entry_point_group="test.group",
            config_base_class=TestTypedFactory.MyConfig,
        )
        cfg = TestTypedFactory.MyConfig(kind="")

        with self.assertRaises(ToDoError) as cm:
            factory.create(cfg)

        msg = str(cm.exception)
        self.assertIn("Attr kind is required in the configuration", msg)

    def test_create_return_instance_when_plugin_accepts_config_positional(self):
        """当插件类 __init__ 接受 (config, *args, **kwargs) 形式时，应按位置参数创建实例"""

        from msmodelslim.utils.plugin import typed_factory as typed_factory_module

        factory = TypedFactory[object](
            entry_point_group="test.group",
            config_base_class=TestTypedFactory.MyConfig,
        )
        cfg = TestTypedFactory.MyConfig(kind="service_a", value=5)

        class ServiceA:
            def __init__(self, config, *args, **kwargs):
                self.config = config
                self.args = args
                self.kwargs = kwargs

        def fake_load(group, plugin_type, base):
            self.assertEqual("test.group", group)
            self.assertEqual("service_a", plugin_type)
            self.assertIs(object, base)
            return ServiceA

        with patch.object(typed_factory_module, "load_plugin_class", fake_load):
            inst = factory.create(cfg, 1, flag=True)

        self.assertIsInstance(inst, ServiceA)
        self.assertIs(cfg, inst.config)
        self.assertEqual((1,), inst.args)
        self.assertEqual({"flag": True}, inst.kwargs)

    def test_create_return_instance_when_plugin_accepts_config_keyword(self):
        """当插件类 __init__ 接受 config 关键字参数时，应按照关键字参数创建实例"""

        from msmodelslim.utils.plugin import typed_factory as typed_factory_module

        factory = TypedFactory[object](
            entry_point_group="test.group",
            config_base_class=TestTypedFactory.MyConfig,
        )
        cfg = TestTypedFactory.MyConfig(kind="service_b", value=7)

        class ServiceB:
            def __init__(self, *args, config: TestTypedFactory.MyConfig, **kwargs):
                self.config = config
                self.args = args
                self.kwargs = kwargs

        def fake_load(group, plugin_type, base):
            self.assertEqual("test.group", group)
            self.assertEqual("service_b", plugin_type)
            self.assertIs(object, base)
            return ServiceB

        with patch.object(typed_factory_module, "load_plugin_class", fake_load):
            inst = factory.create(cfg, "x", y=2)

        self.assertIsInstance(inst, ServiceB)
        self.assertIs(cfg, inst.config)
        self.assertEqual(("x",), inst.args)
        self.assertEqual({"y": 2}, inst.kwargs)

    def test_create_return_instance_when_use_config_model_dump(self):
        """当既不能按位置也不能按关键字传递 config 时，应使用 config.model_dump 展开参数"""

        from msmodelslim.utils.plugin import typed_factory as typed_factory_module

        factory = TypedFactory[object](
            entry_point_group="test.group",
            config_base_class=TestTypedFactory.MyConfig,
        )
        cfg = TestTypedFactory.MyConfig(kind="service_c", value=3)

        class ServiceC:
            def __init__(self, kind: str, value: int, **kwargs):
                self.kind = kind
                self.value = value
                self.kwargs = kwargs

        def fake_load(group, plugin_type, base):
            self.assertEqual("test.group", group)
            self.assertEqual("service_c", plugin_type)
            self.assertIs(object, base)
            return ServiceC

        with patch.object(typed_factory_module, "load_plugin_class", fake_load):
            inst = factory.create(cfg, extra="ok")

        self.assertIsInstance(inst, ServiceC)
        self.assertEqual("service_c", inst.kind)
        self.assertEqual(3, inst.value)
        # kwargs 中应同时包含 config 字段和额外参数
        self.assertEqual("ok", inst.kwargs["extra"])
