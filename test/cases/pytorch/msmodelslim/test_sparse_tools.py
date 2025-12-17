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

import torch
import pytest

try:
    import torch_npu
except Exception as e:
    torch_npu = None

try:
    from msmodelslim.pytorch.sparse import sparse_tools
except Exception as e:
    sparse_tools = None


class TwoLayerNet(torch.nn.Module):
    def __init__(self, D_in, H, D_out):
        super(TwoLayerNet, self).__init__()
        self.linear1 = torch.nn.Linear(D_in, H, bias=True)
        self.linear2 = torch.nn.Linear(H, D_out, bias=True)
    
    def forward(self, x):
        _ = self.linear1(x)
        y_pred = self.linear2(_)
        return y_pred
    

@pytest.fixture(scope="function")
def generate_model():
    model = TwoLayerNet(100, 10, 1)
    yield model


def get_sparsity(model):
    total_params = 0
    total_zero_params = 0
    for name, param in model.named_parameters():
        if 'weight' in name:
            total_params += param.numel()
            total_zero_params += torch.sum(param == 0).item()
    try:
        sparsity = total_zero_params / total_params
    except ZeroDivisionError as error:
        raise ZeroDivisionError("divisor cannot be zero!") from error
    return sparsity


@pytest.mark.skipif(torch_npu is None or sparse_tools is None, reason="requires torch_npu and sparse_tools")
def test_sparse_tools_given_model_then_pass(generate_model):
    # 修改 test_dataset 为一个包含字符串的列表
    test_dataset = ["example_string_1", "example_string_2"]
    sparse_config = sparse_tools.SparseConfig(method='magnitude')
    prune_compressor = sparse_tools.Compressor(generate_model, sparse_config)
    prune_compressor.compress(dataset=test_dataset)