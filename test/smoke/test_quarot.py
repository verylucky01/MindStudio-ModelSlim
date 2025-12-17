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
import os
import shutil
import tempfile

import pytest
import torch

from .base import FakeLlamaModelAdapter, invoke_test, is_npu_available


@pytest.mark.parametrize("test_device, test_dtype", [
    pytest.param("cpu", torch.float32),
    pytest.param("npu", torch.float16, marks=pytest.mark.skipif(not is_npu_available(), reason="NPU not available")),
    pytest.param("npu", torch.bfloat16, marks=pytest.mark.skipif(not is_npu_available(), reason="NPU not available")),
])
@pytest.mark.smoke
def test_quarot_only_process(test_device: str, test_dtype: torch.dtype):
    tmp_dir = tempfile.mkdtemp()

    try:
        # 执行QuaRot量化测试
        model_adapter = invoke_test("quarot_only.yaml", tmp_dir, device=test_device)

        assert isinstance(model_adapter, FakeLlamaModelAdapter), "model_adapter should be FakeLlamaModelAdapter"

        # 测试伪量化
        tokenizer = model_adapter.loaded_tokenizer
        input_text = "Hello world"
        input_ids = tokenizer(input_text, return_tensors="pt", truncation=True)
        model_adapter.loaded_model(**input_ids)

    finally:
        # 清理临时目录
        if os.path.exists(tmp_dir):
            shutil.rmtree(tmp_dir)


@pytest.mark.parametrize("test_device, test_dtype", [
    pytest.param("cpu", torch.float32),
    pytest.param("npu", torch.float16, marks=pytest.mark.skipif(not is_npu_available(), reason="NPU not available")),
    pytest.param("npu", torch.bfloat16, marks=pytest.mark.skipif(not is_npu_available(), reason="NPU not available")),
])
@pytest.mark.smoke
def test_quarot_autoround_process(test_device: str, test_dtype: torch.dtype):
    tmp_dir = tempfile.mkdtemp()

    try:
        # 执行QuaRot+AutoRound量化测试
        model_adapter = invoke_test("quarot_autoround.yaml", tmp_dir, device=test_device)

        assert isinstance(model_adapter, FakeLlamaModelAdapter), "model_adapter should be FakeLlamaModelAdapter"

        # 测试伪量化
        tokenizer = model_adapter.loaded_tokenizer
        input_text = "Hello world"
        input_ids = tokenizer(input_text, return_tensors="pt", truncation=True)
        model_adapter.loaded_model(**input_ids)

    finally:
        # 清理临时目录
        if os.path.exists(tmp_dir):
            shutil.rmtree(tmp_dir)
