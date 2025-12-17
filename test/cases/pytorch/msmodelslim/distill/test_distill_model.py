# Copyright (c) 2023-2024 Huawei Technologies Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

try:
    from torch.tensor import Tensor
except ModuleNotFoundError as ee:
    from torch import Tensor
from resources.sample_net_torch import TorchTeacherModel
from resources.sample_net_torch import TorchStudentModel
import pytest

from msmodelslim.common.knowledge_distill.knowledge_distill import KnowledgeDistillConfig
from msmodelslim.common.knowledge_distill.knowledge_distill import get_distill_model


@pytest.fixture()
def config():
    distill_config = KnowledgeDistillConfig()
    distill_config.set_hard_label(0.5, 0) \
        .add_inter_soft_label({
            "t_module": "teacher_fc",
            "s_module": "student_fc",
            "t_output_idx": 0,
            "s_output_idx": 0,
            "loss_func": [{
                "func_name": "KDCrossEntropy",
                "func_weight": 1
            }]
        }).add_output_soft_label({
            "t_output_idx": 0,
            "s_output_idx": 0,
            "loss_func": [{
                "func_name": "KDCrossEntropy",
                "func_weight": 1                
            }]
        })
    yield distill_config


@pytest.fixture()
def student_model():
    yield TorchStudentModel()


@pytest.fixture()
def teacher_model():
    yield TorchTeacherModel()


class TestDistillModel(object):
    def test_get_distill_model_given_valid_when_any_then_pass(self, config, teacher_model, student_model):
        distill_model = get_distill_model(teacher_model, student_model, config)
        assert distill_model.student_model is student_model
        assert distill_model.teacher_model is teacher_model
    
    def test_get_student_model_given_valid_when_any_then_pass(self, config, teacher_model, student_model):
        distill_model = get_distill_model(teacher_model, student_model, config)
        input_data = Tensor([[1.0]])
        distill_model(input_data)
        distilled_student = distill_model.get_student_model()
        assert distilled_student is student_model