# Copyright Huawei Technologies Co., Ltd. 2022-2022. All rights reserved.

import torch
import mindspore.nn as nn


class TorchPrunedModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(1, 1)

    def forward(self, inputs):
        return self.fc1(inputs)


class TorchOriModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(2, 2)
        self.fc2 = torch.nn.Linear(2, 2)

    def forward(self, inputs):
        output = self.fc1(inputs)
        output = self.fc2(output)
        return output


class MsPrunedModel(nn.Cell):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Dense(1, 1)

    def construct(self, *inputs, **kwargs):
        return self.fc1(*inputs)


class MsOriModel(nn.Cell):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Dense(2, 2)
        self.fc2 = nn.Dense(2, 2)

    def construct(self, *inputs, **kwargs):
        output = self.fc1(*inputs)
        output = self.fc2(output)
        return output