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
import pytest

from ascend_utils.pytorch.knowledge_distill.distill_losses_func_torch import DISTILL_LOSS_FUNC_TORCH


@pytest.fixture()
def tensor_s():
    yield Tensor([[1.0], [1.0]])


@pytest.fixture()
def tensor_t():
    yield Tensor([[1.0], [1.0]])


def test_kd_cross_entropy_given_valid_when_any_then_pass(tensor_s, tensor_t):
    loss_func_kd_cross_entropy = DISTILL_LOSS_FUNC_TORCH["KDCrossEntropy"]()
    loss_func_kd_cross_entropy(tensor_s, tensor_t)