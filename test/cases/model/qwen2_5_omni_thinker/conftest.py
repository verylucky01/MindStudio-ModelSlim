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

"""
Pytest config for qwen2_5_omni_thinker tests.
"""

import sys
import types
import importlib
from unittest.mock import MagicMock

_created_modules = {}
_original_modules = {}


def _setup_mock_modules():
    """Inject mock modules for qwen2_5_omni_thinker so tests run without full transformers stack."""
    import transformers  # ensure loaded before we store reference

    _original_modules["transformers"] = sys.modules["transformers"]
    transformers_module = sys.modules["transformers"]

    required_attrs = {
        "Qwen2_5OmniThinkerForConditionalGeneration": MagicMock(),
        "Qwen2_5OmniProcessor": MagicMock(),
    }

    for attr_name, attr_value in required_attrs.items():
        if not hasattr(transformers_module, attr_name):
            setattr(transformers_module, attr_name, attr_value)

    if "transformers.models" not in sys.modules:
        try:
            importlib.import_module("transformers.models")
        except Exception:
            pass

    if "transformers.models" in sys.modules:
        _original_modules["transformers.models"] = sys.modules["transformers.models"]
        models_module = sys.modules["transformers.models"]
    else:
        _original_modules["transformers.models"] = None
        models_module = types.SimpleNamespace()
        setattr(transformers_module, "models", models_module)

    if "transformers.models.qwen2_5_omni" not in sys.modules:
        _original_modules["transformers.models.qwen2_5_omni"] = None
        qwen2_5_omni_module = types.ModuleType("transformers.models.qwen2_5_omni")
        sys.modules["transformers.models.qwen2_5_omni"] = qwen2_5_omni_module
        setattr(models_module, "qwen2_5_omni", qwen2_5_omni_module)
        _created_modules["transformers.models.qwen2_5_omni"] = qwen2_5_omni_module
    else:
        _original_modules["transformers.models.qwen2_5_omni"] = sys.modules[
            "transformers.models.qwen2_5_omni"
        ]
        qwen2_5_omni_module = sys.modules["transformers.models.qwen2_5_omni"]

    if "transformers.models.qwen2_5_omni.modeling_qwen2_5_omni" not in sys.modules:
        def _make_modeling():
            m = types.ModuleType(
                "transformers.models.qwen2_5_omni.modeling_qwen2_5_omni"
            )

            class MockQwen2_5OmniDecoderLayer:
                pass

            m.Qwen2_5OmniDecoderLayer = MockQwen2_5OmniDecoderLayer
            return m

        _original_modules["transformers.models.qwen2_5_omni.modeling_qwen2_5_omni"] = (
            None
        )
        mock_modeling = _make_modeling()
        sys.modules["transformers.models.qwen2_5_omni.modeling_qwen2_5_omni"] = (
            mock_modeling
        )
        setattr(qwen2_5_omni_module, "modeling_qwen2_5_omni", mock_modeling)
        _created_modules["transformers.models.qwen2_5_omni.modeling_qwen2_5_omni"] = (
            mock_modeling
        )
    else:
        _original_modules["transformers.models.qwen2_5_omni.modeling_qwen2_5_omni"] = (
            sys.modules["transformers.models.qwen2_5_omni.modeling_qwen2_5_omni"]
        )

    if "transformers.masking_utils" not in sys.modules:
        _original_modules["transformers.masking_utils"] = None
        masking_utils_module = types.ModuleType("transformers.masking_utils")
        masking_utils_module.create_causal_mask = MagicMock()
        masking_utils_module.create_sliding_window_causal_mask = MagicMock()
        sys.modules["transformers.masking_utils"] = masking_utils_module
        setattr(transformers_module, "masking_utils", masking_utils_module)
        _created_modules["transformers.masking_utils"] = masking_utils_module
    else:
        _original_modules["transformers.masking_utils"] = sys.modules[
            "transformers.masking_utils"
        ]

    if "qwen_omni_utils" not in sys.modules:
        _original_modules["qwen_omni_utils"] = None
        qwen_omni_utils = types.ModuleType("qwen_omni_utils")
        sys.modules["qwen_omni_utils"] = qwen_omni_utils
        _created_modules["qwen_omni_utils"] = qwen_omni_utils
    else:
        _original_modules["qwen_omni_utils"] = sys.modules["qwen_omni_utils"]
        qwen_omni_utils = sys.modules["qwen_omni_utils"]

    if not hasattr(qwen_omni_utils, "process_mm_info"):
        qwen_omni_utils.process_mm_info = MagicMock()


_setup_mock_modules()


def pytest_configure(config):
    """Ensure mocks are in place before test modules are imported."""
    _setup_mock_modules()


def pytest_unconfigure(config):
    """Remove only modules created by this conftest; restore originals when present."""
    for module_name in _created_modules:
        if module_name in sys.modules:
            del sys.modules[module_name]
        if _original_modules.get(module_name) is not None:
            sys.modules[module_name] = _original_modules[module_name]
