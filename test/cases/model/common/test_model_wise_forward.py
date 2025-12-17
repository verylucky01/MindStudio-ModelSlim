# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
import unittest
from typing import Tuple, Dict, Any

import torch
import torch.nn as nn

from msmodelslim.model.common.model_wise_forward import (
    model_wise_forward_func,
    model_wise_visit_func,
)
from msmodelslim.core.base.protocol import ProcessRequest


class DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


class TestModelWiseForward(unittest.TestCase):

    def setUp(self):
        self.model = DummyModel()
        torch.manual_seed(0)

    def test_forward_func_with_list_inputs(self):
        inputs = [torch.randn(2, 4)]
        gen = model_wise_forward_func(self.model, inputs)
        req = self._collect(gen)
        self.assertEqual(req.name, "")
        self.assertIs(req.module, self.model)
        self.assertIsInstance(req.args, list)
        self.assertEqual(len(req.args), 1)
        self.assertIs(req.args[0], inputs[0])
        self.assertEqual(req.kwargs, {})

    def test_forward_func_with_tuple_inputs(self):
        inputs = (torch.randn(2, 4),)
        gen = model_wise_forward_func(self.model, inputs)
        req = self._collect(gen)
        self.assertIs(req.args[0], inputs[0])

    def test_forward_func_with_dict_inputs(self):
        inputs: Dict[str, Any] = {"x": torch.randn(2, 4)}
        gen = model_wise_forward_func(self.model, inputs)
        req = self._collect(gen)
        self.assertEqual(req.args, [inputs])
        self.assertEqual(req.kwargs, inputs)

    def test_visit_func(self):
        gen = model_wise_visit_func(self.model)
        req = next(gen)
        self.assertEqual(req.name, "")
        self.assertIs(req.module, self.model)
        self.assertEqual(req.args, tuple())
        self.assertEqual(req.kwargs, {})

    def _collect(self, gen) -> ProcessRequest:
        req = next(gen)
        self.assertIsInstance(req, ProcessRequest)
        return req
