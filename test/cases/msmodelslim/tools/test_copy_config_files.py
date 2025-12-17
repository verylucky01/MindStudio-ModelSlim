# Copyright Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
import os
import tempfile
import shutil
import json
import pytest

from unittest.mock import MagicMock

from msmodelslim.tools import copy_config_files

def make_json_file(path, content):
    with open(path, "w") as f:
        json.dump(content, f)

def test_copy_json(monkeypatch, tmp_path):
    src = tmp_path / "src.json"
    dst = tmp_path / "dst.json"
    src.write_text('{"a":1}')
    called = {}
    def fake_copy(src_path, dst_path):
        called['copied'] = (src_path, dst_path)
    monkeypatch.setattr(copy_config_files, "safe_copy_file", fake_copy)
    copy_config_files.copy_json(str(src), str(dst), None, False)
    assert called['copied'] == (str(src), str(dst))

def test_modify_config_json_mindie(monkeypatch, tmp_path):
    src = tmp_path / "config.json"
    dst = tmp_path / "out.json"
    quant_desc = tmp_path / "quant_model_description1.json"
    make_json_file(src, {"a": 1})
    make_json_file(quant_desc, {"desc": 1})
    quant_config = MagicMock()
    quant_config.model_quant_type.value = "W8A8"
    quant_config.use_kvcache_quant = True
    quant_config.use_fa_quant = True
    quant_config.group_size = 64

    # mindie_format=True
    def fake_glob(pattern):
        return [str(quant_desc)]
    monkeypatch.setattr(copy_config_files.glob, "glob", lambda pattern: [str(quant_desc)])
    monkeypatch.setattr(copy_config_files, "json_safe_load", lambda path, **kwargs: json.load(open(path)))
    monkeypatch.setattr(copy_config_files, "json_safe_dump", lambda data, path, indent=4: open(path, "w").write(json.dumps(data)))
    monkeypatch.setattr(copy_config_files, "get_valid_write_path", lambda path, is_dir=False: path)

    copy_config_files.modify_config_json(str(src), str(dst), quant_config, mindie_format=True)
    with open(dst) as f:
        out = json.load(f)
    assert "quantization_config" in out

def test_modify_vllm_config_json(monkeypatch, tmp_path):
    src = tmp_path / "config.json"
    dst = tmp_path / "out.json"
    make_json_file(src, {"a": 1})
    quant_config = MagicMock()
    monkeypatch.setattr(copy_config_files, "json_safe_load", lambda path, **kwargs: json.load(open(path)))
    monkeypatch.setattr(copy_config_files, "json_safe_dump", lambda data, path, indent=4: open(path, "w").write(json.dumps(data)))
    copy_config_files.modify_vllm_config_json(str(src), str(dst), quant_config, mindie_format=False)
    with open(dst) as f:
        out = json.load(f)
    assert out["a"] == 1

def test_copy_config_files(monkeypatch, tmp_path):
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    output_dir.mkdir()
    # .json, .py, .index.json, 其它
    (input_dir / "config.json").write_text("{}")
    (input_dir / "other.json").write_text("{}")
    (input_dir / "skip.index.json").write_text("{}")
    (input_dir / "script.py").write_text("# test")
    (input_dir / "notcopy.txt").write_text("no")
    quant_config = MagicMock()
    quant_config.model_quant_type.value = "W8A8"
    quant_config.use_kvcache_quant = False
    quant_config.use_fa_quant = False
    quant_config.group_size = 32

    called = {}
    def fake_get_valid_read_path(path, extensions=None):
        return path
    def fake_get_valid_write_path(path):
        return path
    def fake_copy_json(src, dst, quant_config, mindie_format):
        called[(src, dst)] = "copy_json"
    monkeypatch.setattr(copy_config_files, "get_valid_read_path", fake_get_valid_read_path)
    monkeypatch.setattr(copy_config_files, "get_valid_write_path", fake_get_valid_write_path)
    monkeypatch.setattr(copy_config_files, "copy_json", fake_copy_json)
    monkeypatch.setattr(copy_config_files, "modify_config_json", fake_copy_json)
    monkeypatch.setattr(copy_config_files, "modify_vllm_config_json", fake_copy_json)
    monkeypatch.setattr(copy_config_files, "DEFAULT_FILE_HOOKS", fake_copy_json)
    monkeypatch.setattr(copy_config_files, "FILE_HOOKS", {"config.json": fake_copy_json})

    copy_config_files.copy_config_files(str(input_dir), str(output_dir), quant_config)
    # 检查只处理了.json和.py文件，且跳过了index.json和notcopy.txt
    assert (str(input_dir / "config.json"), str(output_dir / "config.json")) in called
    assert (str(input_dir / "other.json"), str(output_dir / "other.json")) in called
    assert (str(input_dir / "script.py"), str(output_dir / "script.py")) in called
    assert (str(input_dir / "skip.index.json"), str(output_dir / "skip.index.json")) not in called
    assert (str(input_dir / "notcopy.txt"), str(output_dir / "notcopy.txt")) not in called
