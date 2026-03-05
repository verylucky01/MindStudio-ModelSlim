#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""
-------------------------------------------------------------------------
This file is part of the MindStudio project.
Copyright (c) 2026 Huawei Technologies Co.,Ltd.

MindStudio is licensed under Mulan PSL v2.
You can use this software according to the terms and conditions of the Mulan PSL v2.
You may obtain a copy of Mulan PSL v2 at:

         http://license.coscl.org.cn/MulanPSL2

THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
See the Mulan PSL v2 for more details.
-------------------------------------------------------------------------
"""
import os
import shutil
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest
import torch

from .base import FakeLlamaModelAdapter, invoke_test, is_npu_available


def _lab_calib_dir():
    """项目根目录下的 lab_calib，供冒烟测试解析 test.json 等数据集。"""
    root = Path(__file__).resolve().parents[2]
    return root / "lab_calib"


@pytest.mark.parametrize("test_device, test_dtype", [
    pytest.param("cpu", torch.float32),
    pytest.param("npu", torch.float16, marks=pytest.mark.skipif(not is_npu_available(), reason="NPU not available")),
    pytest.param("npu", torch.bfloat16, marks=pytest.mark.skipif(not is_npu_available(), reason="NPU not available")),
])
@pytest.mark.smoke
def test_adapt_rotation_only_process(test_device: str, test_dtype: torch.dtype):
    """AdaptRotation 冒烟：prior 阶段 stage1 优化旋转矩阵，主阶段 stage2 应用旋转，流程可跑通（不涉及保存，与 iter_smooth/kv_quant 一致）."""
    tmp_dir = tempfile.mkdtemp()
    lab_calib = _lab_calib_dir()

    try:
        with patch("msmodelslim.cli.naive_quantization.__main__.get_dataset_dir", return_value=lab_calib):
            model_adapter = invoke_test("adapt_rotation.yaml", tmp_dir, device=test_device)

        assert isinstance(model_adapter, FakeLlamaModelAdapter), "model_adapter should be FakeLlamaModelAdapter"

        # 前向一次确认模型可推理
        tokenizer = model_adapter.loaded_tokenizer
        input_text = "Hello world"
        input_ids = tokenizer(input_text, return_tensors="pt", truncation=True)
        model_adapter.loaded_model(**input_ids)

    finally:
        if os.path.exists(tmp_dir):
            shutil.rmtree(tmp_dir)
