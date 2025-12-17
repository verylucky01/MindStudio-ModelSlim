# Copyright Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
import os
import tempfile
import torch
import pytest

from msmodelslim.tools import add_safetensors

def test_calculate_tensor_size():
    t = torch.ones(3, 4, dtype=torch.float32)
    assert add_safetensors.calculate_tensor_size(t) == 3 * 4 * 4

def test_get_prefix():
    assert add_safetensors.get_prefix("a.b.c.d", -1) == "a.b.c"
    assert add_safetensors.get_prefix("a.b.c.d", 2) == "a.b"

def test_find_file_with_pattern(tmp_path):
    f = tmp_path / "test.index.json"
    f.write_text("{}")
    found = add_safetensors.find_file_with_pattern(str(tmp_path), "*.index.json")
    assert found.endswith("test.index.json")
    # 测试找不到
    with pytest.raises(FileNotFoundError):
        add_safetensors.find_file_with_pattern(str(tmp_path), "*.notfound.json")
    # 测试多个文件
    (tmp_path / "test2.index.json").write_text("{}")
    with pytest.raises(ValueError):
        add_safetensors.find_file_with_pattern(str(tmp_path), "*.index.json")

def test_get_weight_map(monkeypatch):
    monkeypatch.setattr(add_safetensors, "json_safe_load", lambda path: {"weight_map": {"a": "b"}})
    assert add_safetensors.get_weight_map("dummy.json") == {"a": "b"}

def test_get_tensor(monkeypatch, tmp_path):
    class DummySafeOpen:
        def __init__(self, *a, **k): pass
        def __enter__(self):
            class F:
                def keys(self): return ["a"]
                def get_tensor(self, name): return torch.ones(2, 2)
            return F()
        def __exit__(self, exc_type, exc_val, exc_tb): pass
    monkeypatch.setattr(add_safetensors, "safe_open", DummySafeOpen)
    weight_map = {"a": "file1"}
    safetensor_path = str(tmp_path)
    (tmp_path / "file1").write_text("dummy")
    t = add_safetensors.get_tensor("a", safetensor_path, weight_map)
    assert isinstance(t, torch.Tensor)
    # 测试tensor不存在
    class DummySafeOpen2:
        def __init__(self, *a, **k): pass
        def __enter__(self):
            class F:
                def keys(self): return []
                def get_tensor(self, name): return None
            return F()
        def __exit__(self, exc_type, exc_val, exc_tb): pass
    monkeypatch.setattr(add_safetensors, "safe_open", DummySafeOpen2)
    with pytest.raises(KeyError):
        add_safetensors.get_tensor("notfound", safetensor_path, weight_map={"notfound": "file1"})

def test_add_safetensors_main(monkeypatch, tmp_path):
    org_dir = tmp_path / "org"
    quant_dir = tmp_path / "quant"
    org_dir.mkdir()
    quant_dir.mkdir()
    (quant_dir / "quant_model_weight_1.index.json").write_text('{"weight_map": {}, "metadata": {}}')
    (quant_dir / "quant_model_description_1.json").write_text('{}')
    (org_dir / "float.index.json").write_text('{"weight_map": {"a": "a_file", "b": "b_file", "c.weight_scale_inv": "c_file"}}')
    (org_dir / "a_file").write_text("dummy")
    (org_dir / "b_file").write_text("dummy")
    (org_dir / "c_file").write_text("dummy")
    monkeypatch.setattr(add_safetensors, "get_valid_read_path", lambda path, **k: str(path))
    monkeypatch.setattr(add_safetensors, "find_file_with_pattern", lambda d, p: \
                                str(list((quant_dir if "quant" in str(d) else org_dir).glob("*.json"))[0]))
    monkeypatch.setattr(add_safetensors, "json_safe_load", lambda path: {"weight_map": {}, "metadata": {}} if "index" in path else {})
    monkeypatch.setattr(add_safetensors, "json_safe_dump", lambda data, path, indent=4: None)
    class DummySafeOpen:
        def __init__(self, *a, **k): pass
        def __enter__(self):
            class F:
                def keys(self): return ["a", "b", "c.weight_scale_inv"]
                def get_tensor(self, name): return torch.ones(2, 2)
            return F()
        def __exit__(self, exc_type, exc_val, exc_tb): pass
    monkeypatch.setattr(add_safetensors, "safe_open", DummySafeOpen)
    monkeypatch.setattr(add_safetensors, "save_file", lambda data, path: None)
    monkeypatch.setattr(add_safetensors, "weight_dequant", lambda t, s: t)
    monkeypatch.setattr(add_safetensors, "msmodelslim_logger", \
                            type("L", (), {"info": staticmethod(print), "warning": staticmethod(print)})())
    add_safetensors.add_safetensors(str(org_dir), str(quant_dir), "mtp", max_file_size_gb=0.00001)

def test_add_safetensors_weight_scale_inv_missing(monkeypatch, tmp_path):
    org_dir = tmp_path / "org"
    quant_dir = tmp_path / "quant"
    org_dir.mkdir()
    quant_dir.mkdir()
    (quant_dir / "quant_model_weight_1.index.json").write_text('{"weight_map": {}, "metadata": {}}')
    (quant_dir / "quant_model_description_1.json").write_text('{}')
    (org_dir / "float.index.json").write_text('{"weight_map": {"a": "a_file"}}')
    (org_dir / "a_file").write_text("dummy")
    monkeypatch.setattr(add_safetensors, "get_valid_read_path", lambda path, **k: str(path))
    monkeypatch.setattr(add_safetensors, "find_file_with_pattern", lambda d, p: \
                                str(list((quant_dir if "quant" in str(d) else org_dir).glob("*.json"))[0]))
    monkeypatch.setattr(add_safetensors, "json_safe_load", \
                                        lambda path: {"weight_map": {}, "metadata": {}} if "index" in path else {})
    monkeypatch.setattr(add_safetensors, "json_safe_dump", lambda data, path, indent=4: None)
    class DummySafeOpen:
        def __init__(self, *a, **k): pass
        def __enter__(self):
            class F:
                def keys(self): return ["a"]
                def get_tensor(self, name): return torch.ones(2, 2)
            return F()
        def __exit__(self, exc_type, exc_val, exc_tb): pass
    monkeypatch.setattr(add_safetensors, "safe_open", DummySafeOpen)
    monkeypatch.setattr(add_safetensors, "save_file", lambda data, path: None)
    monkeypatch.setattr(add_safetensors, "weight_dequant", lambda t, s: t)
    monkeypatch.setattr(add_safetensors, "msmodelslim_logger", \
                            type("L", (), {"info": staticmethod(print), "warning": staticmethod(print)})())
    add_safetensors.add_safetensors(str(org_dir), str(quant_dir), "mtp", max_file_size_gb=5)
