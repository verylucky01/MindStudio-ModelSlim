# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
import unittest
from pathlib import Path

from msmodelslim.model.base import BaseModelAdapter


class TestBaseModelAdapter(unittest.TestCase):

    def setUp(self):
        self.model_type = "Qwen-7B"
        self.model_path = Path("/tmp/model")
        self.trust_remote_code = True

    def test_init_and_properties(self):
        adapter = BaseModelAdapter(
            model_type=self.model_type,
            model_path=self.model_path,
            trust_remote_code=self.trust_remote_code,
        )

        self.assertEqual(adapter.model_type, self.model_type)
        self.assertEqual(adapter.model_path, self.model_path)
        self.assertEqual(adapter.trust_remote_code, self.trust_remote_code)

    def test_setters_update_values(self):
        adapter = BaseModelAdapter(
            model_type=self.model_type,
            model_path=self.model_path,
            trust_remote_code=self.trust_remote_code,
        )

        new_model_type = "Qwen-14B"
        new_model_path = Path("/opt/models/qwen")
        new_trust_remote_code = False

        adapter.model_type = new_model_type
        adapter.model_path = new_model_path
        adapter.trust_remote_code = new_trust_remote_code

        self.assertEqual(adapter.model_type, new_model_type)
        self.assertEqual(adapter.model_path, new_model_path)
        self.assertEqual(adapter.trust_remote_code, new_trust_remote_code)
