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

import pytest

from msmodelslim.core.quant_service.multimodal_sd_v1.quant_service import MultimodalPipelineInterface
from msmodelslim.utils.exception import ToDoError


# 定义一个完全实现所有抽象方法的基础测试类
class FullImplementedBase(MultimodalPipelineInterface):
    """完全实现所有抽象方法的基础类，用于测试"""

    # 实现IModel的抽象属性
    model_path = "mock_path"
    model_type = "mock_type"
    trust_remote_code = False

    # 实现PipelineInterface的抽象方法
    def handle_dataset(self, dataset, device=None):
        return []

    def init_model(self, device=None):
        return None  # 实际项目中是nn.Module，测试用None即可

    def generate_model_visit(self, model):
        # 生成器必须返回至少一个值，用空对象模拟
        yield object()

    def generate_model_forward(self, model, inputs):
        yield object()  # 同上

    def enable_kv_cache(self, model, need_kv_cache):
        pass  # 空实现

    # 实现MultimodalPipelineInterface的抽象方法（空实现，测试时会被重写）
    def run_calib_inference(self):
        pass

    def apply_quantization(self, quant_model_func):
        pass

    def load_pipeline(self):
        pass

    def set_model_args(self, override_model_config):
        pass


class TestMultimodalPipelineExceptions:
    """测试MultimodalPipelineInterface的异常抛出"""

    def test_run_calib_inference_throws_todo_error(self):
        """测试run_calib_inference未实现时的异常"""

        class TestSubclass(FullImplementedBase):
            # 重写方法，调用父类（MultimodalPipelineInterface）的实现
            def run_calib_inference(self):
                super(FullImplementedBase, self).run_calib_inference()

        # 实例化子类（此时不会报错，因为所有抽象方法都已实现）
        instance = TestSubclass()

        # 验证调用时抛出异常
        with pytest.raises(ToDoError) as exc_info:
            instance.run_calib_inference()

        assert "This model does not support run_calib_inference." in str(exc_info.value)

    def test_apply_quantization_throws_todo_error(self):
        """测试apply_quantization未实现时的异常"""

        class TestSubclass(FullImplementedBase):
            def apply_quantization(self, quant_model_func):
                super(FullImplementedBase, self).apply_quantization(quant_model_func)

        instance = TestSubclass()
        with pytest.raises(ToDoError) as exc_info:
            instance.apply_quantization(None)

        assert "This model does not support apply_quantization." in str(exc_info.value)

    def test_load_pipeline_throws_todo_error(self):
        """测试load_pipeline未实现时的异常"""

        class TestSubclass(FullImplementedBase):
            def load_pipeline(self):
                super(FullImplementedBase, self).load_pipeline()

        instance = TestSubclass()
        with pytest.raises(ToDoError) as exc_info:
            instance.load_pipeline()

        assert "This model does not support load_pipeline." in str(exc_info.value)

    def test_set_model_args_throws_todo_error(self):
        """测试set_model_args未实现时的异常"""

        class TestSubclass(FullImplementedBase):
            def set_model_args(self, override_model_config):
                super(FullImplementedBase, self).set_model_args(override_model_config)

        instance = TestSubclass()
        with pytest.raises(ToDoError) as exc_info:
            instance.set_model_args(None)

        assert "This model does not support set_model_args." in str(exc_info.value)
