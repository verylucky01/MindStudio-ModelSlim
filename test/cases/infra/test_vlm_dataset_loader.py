#! /usr/bin/env python3
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

"""
Unit tests for `msmodelslim.infra.vlm_dataset_loader.VLMDatasetLoader`.

这些用例主要覆盖：
- 路径解析逻辑（绝对路径、相对路径、dataset_dir 组合以及不存在路径）。
- 四类数据类型：纯文本 json/jsonl、纯图片（默认文案）、图片+自定义文案、混合文本+图片。
- 异常场景：不支持的路径类型、空数据、解析失败等。
"""

from pathlib import Path
from typing import List
from unittest.mock import patch

import json
import pytest

from msmodelslim.infra.vlm_dataset_loader import (
    VLMDatasetLoader,
    VlmCalibSample,
)
from msmodelslim.utils.exception import InvalidDatasetError


@pytest.fixture(autouse=True)
def mock_get_valid_read_path():
    """
    自动 mock 掉安全路径检查，避免 UT 依赖环境权限/安全策略。

    这里直接返回传入的路径字符串，保持最小行为假设。
    """

    def _fake_get_valid_read_path(path: str, *_, **__) -> str:
        return path

    with patch(
        "msmodelslim.infra.vlm_dataset_loader.get_valid_read_path",
        side_effect=_fake_get_valid_read_path,
    ):
        yield


def _write_jsonl(path: Path, lines: List[object]) -> None:
    """便捷工具：按行写入 JSONL 文件。"""
    with path.open("w", encoding="utf-8") as f:
        for item in lines:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def test_get_dataset_by_name_pure_text_jsonl(tmp_path: Path):
    """
    验证纯文本 JSONL 文件的加载逻辑（Type 1）。

    - 行可以是字符串或包含 text 字段的字典
    - 无效行（缺少 text 字段或 JSON 解析失败）会被跳过
    """
    jsonl_path = tmp_path / "calib.jsonl"
    _write_jsonl(
        jsonl_path,
        [
            "who are you?",  # 纯字符串
            {"text": "hello"},  # 合法字典
            {"invalid": "ignored"},  # 无效字典
        ],
    )

    loader = VLMDatasetLoader()
    dataset = loader.get_dataset_by_name(str(jsonl_path))

    assert len(dataset) == 2
    assert all(isinstance(s, VlmCalibSample) for s in dataset)
    assert dataset[0].text == "who are you?"
    assert dataset[0].image is None
    assert dataset[1].text == "hello"
    assert dataset[1].image is None


def test_get_dataset_by_name_uses_dataset_dir_when_relative_not_exists(tmp_path: Path):
    """
    覆盖 get_dataset_by_name 使用 dataset_dir 组合相对路径的逻辑（120-129 分支）。
    - 当前工作目录下文件不存在
    - 但在 dataset_dir 下存在对应 jsonl 文件
    """
    calib_dir = tmp_path / "lab_calib"
    calib_dir.mkdir()
    jsonl_path = calib_dir / "short.jsonl"
    _write_jsonl(jsonl_path, ["hello from dataset_dir"])

    loader = VLMDatasetLoader(dataset_dir=calib_dir)
    dataset = loader.get_dataset_by_name("short.jsonl")

    assert len(dataset) == 1
    assert dataset[0].text == "hello from dataset_dir"


def test_get_dataset_by_name_directory_pure_images_with_default_text(tmp_path: Path):
    """
    验证仅包含图片文件的目录（无 json/jsonl）会走 Type 2：
    - 使用 default_text 生成样本
    """
    img1 = tmp_path / "a.jpg"
    img2 = tmp_path / "b.png"
    img1.write_bytes(b"dummy")
    img2.write_bytes(b"dummy")

    loader = VLMDatasetLoader(default_text="DEFAULT_PROMPT")
    dataset = loader.get_dataset_by_name(str(tmp_path))

    assert len(dataset) == 2
    texts = {s.text for s in dataset}
    images = {Path(s.image).name for s in dataset}
    assert texts == {"DEFAULT_PROMPT"}
    assert images == {"a.jpg", "b.png"}


def test_get_dataset_by_name_directory_with_mixed_jsonl(tmp_path: Path):
    """
    验证包含图片与 JSONL 的目录会走 Type 3/4：
    - JSONL 中既有纯文本，也有 image+text。
    """
    # 准备图片
    img1 = tmp_path / "img1.jpg"
    img2 = tmp_path / "img2.jpg"
    img1.write_bytes(b"1")
    img2.write_bytes(b"2")

    # 准备 JSONL：一条文本、一条 image+text
    jsonl_path = tmp_path / "data.jsonl"
    _write_jsonl(
        jsonl_path,
        [
            {"text": "only text"},
            {"image": "img1.jpg", "text": "image text"},
        ],
    )

    loader = VLMDatasetLoader(default_text="PROMPT")
    dataset = loader.get_dataset_by_name(str(tmp_path))

    # 期望：两条样本：1 条文本-only + 1 条 image+text
    assert len(dataset) == 2
    assert any(s.image is None and s.text == "only text" for s in dataset)
    assert any(
        s.image is not None and Path(s.image).name == "img1.jpg" and s.text == "image text"
        for s in dataset
    )


def test_get_dataset_by_name_unsupported_path_type_raises_invalid_dataset_error(tmp_path: Path):
    """
    验证当路径存在但既不是目录也不是 json/jsonl 文件时，会抛出 InvalidDatasetError。
    """
    txt = tmp_path / "not_supported.txt"
    txt.write_text("dummy", encoding="utf-8")

    loader = VLMDatasetLoader()
    with pytest.raises(InvalidDatasetError) as exc_info:
        loader.get_dataset_by_name(str(txt))

    assert "not a valid type" in str(exc_info.value)


def test_get_dataset_by_name_path_not_exists_raises_invalid_dataset_error(tmp_path: Path):
    """
    验证当最终解析路径不存在时，会抛出 InvalidDatasetError。
    """
    missing = tmp_path / "missing.jsonl"
    loader = VLMDatasetLoader()

    with pytest.raises(InvalidDatasetError) as exc_info:
        loader.get_dataset_by_name(str(missing))

    assert "does not exist" in str(exc_info.value)


def test_get_dataset_by_name_with_dataset_dir_and_fallback_raises_invalid_dataset_error(tmp_path: Path):
    """
    覆盖 get_dataset_by_name 中 dataset_dir 组合后仍不存在、触发
    `_resolve_path_with_fallback` 的场景（120-137、176-185 分支）。
    """
    # dataset_dir 指向一个不存在的目录
    dataset_dir = tmp_path / "lab_calib_not_exist"
    loader = VLMDatasetLoader(dataset_dir=dataset_dir)

    with pytest.raises(InvalidDatasetError) as exc_info:
        loader.get_dataset_by_name("non_exist.jsonl")

    # 来自 _resolve_path_with_fallback 的错误消息
    assert "does not exist" in str(exc_info.value)


def test_resolve_path_with_fallback_general_exception(tmp_path: Path, monkeypatch):
    """
    直接调用 `_resolve_path_with_fallback`，模拟 Path.resolve 抛出通用异常，
    覆盖 188-193 分支。
    """

    class FakePath:
        def resolve(self):
            raise RuntimeError("resolve failed")

    loader = VLMDatasetLoader()

    with pytest.raises(InvalidDatasetError) as exc_info:
        loader._resolve_path_with_fallback(
            "broken", FakePath(), "hint for broken path"
        )

    assert "Failed to resolve dataset path" in str(exc_info.value)


def test_load_text_from_file_json_empty_raises_invalid_dataset_error(tmp_path: Path):
    """
    验证 _load_text_from_file 处理 .json 文件：
    - 当没有任何合法的 text 项时会抛出 InvalidDatasetError。
    """
    json_path = tmp_path / "empty.json"
    # 写入一个空列表，意味着没有有效条目
    json_path.write_text("[]", encoding="utf-8")

    loader = VLMDatasetLoader()
    with pytest.raises(InvalidDatasetError) as exc_info:
        # 通过直接调用私有方法，触发 json 分支逻辑
        loader._load_text_from_file(json_path, file_suffix=".json")

    assert "No valid text entries" in str(exc_info.value)


def test_load_text_from_file_json_decode_error_and_root_warning(tmp_path: Path):
    """
    覆盖 `_load_text_from_file` 中：
    - JSONDecodeError 分支（253-257）
    - Root 级别无效条目触发的 warning 分支（264-272）
    """
    # 1) JSONDecodeError：写入非法 JSON
    bad_json = tmp_path / "bad.json"
    bad_json.write_text("{ invalid json", encoding="utf-8")

    loader = VLMDatasetLoader()
    with pytest.raises(InvalidDatasetError) as exc_info:
        loader._load_text_from_file(bad_json, file_suffix=".json")
    assert "Failed to parse JSON file" in str(exc_info.value)

    # 2) Root 无效条目：写入合法 JSON，但结构不含 text
    root_invalid = tmp_path / "root_invalid.json"
    root_invalid.write_text(json.dumps({"foo": "bar"}), encoding="utf-8")

    with pytest.raises(InvalidDatasetError) as exc_info2:
        loader._load_text_from_file(root_invalid, file_suffix=".json")
    # 仍然会走到 "No valid text entries" 的错误
    assert "No valid text entries" in str(exc_info2.value)


def test_load_text_from_file_jsonl_decode_and_generic_error(tmp_path: Path, monkeypatch):
    """
    覆盖 `_load_text_from_file` JSONL 分支中：
    - json.JSONDecodeError 捕获（242-244）
    - 通用异常捕获（245-247）
    """
    jsonl_path = tmp_path / "bad_lines.jsonl"
    # 第一行非法 JSON，第二行我们通过 monkeypatch 让 json.loads 抛出 RuntimeError
    jsonl_path.write_text('{"bad": "json"\n{"ok": "but will raise"}', encoding="utf-8")

    loader = VLMDatasetLoader()

    original_loads = json.loads

    def fake_loads(s):
        # 第一行：保持原行为，产生 JSONDecodeError
        if "bad" in s:
            return original_loads(s)
        # 第二行：触发通用异常分支
        raise RuntimeError("generic error in json.loads")

    with patch("msmodelslim.infra.vlm_dataset_loader.json.loads", side_effect=fake_loads):
        # 不关心返回值，只要函数能安全执行到结束（所有异常被内部捕获并跳过）
        with pytest.raises(InvalidDatasetError):
            # 由于两行都无法产生有效样本，最终会抛出 "No valid text entries" 错误
            loader._load_text_from_file(jsonl_path, file_suffix=".jsonl")


def test_load_from_directory_no_images_or_data_raises_invalid_dataset_error(tmp_path: Path):
    """
    验证 _load_from_directory 在目录中既无图片也无 json/jsonl 时抛出 InvalidDatasetError。
    """
    # 空目录
    loader = VLMDatasetLoader()
    with pytest.raises(InvalidDatasetError) as exc_info:
        loader._load_from_directory(tmp_path)

    assert "No images or data files found in directory" in str(exc_info.value)


def test_load_mixed_from_jsonl_json_mode_type3_and_type4_and_no_valid(tmp_path: Path):
    """
    覆盖 `_load_mixed_from_jsonl` 的 .json 分支（411-480）：
    - Type3：只有 image+text（text_only_count == 0），且触发图片数量不匹配 warning。
    - Type4：混合 image+text 与 text-only。
    - 无任何有效条目时抛出 InvalidDatasetError。
    """
    loader = VLMDatasetLoader(default_text="DEFAULT_TEXT")

    base_dir = tmp_path
    img1 = base_dir / "img1.jpg"
    img2 = base_dir / "img2.jpg"
    img1.write_bytes(b"1")
    img2.write_bytes(b"2")
    images = [img1, img2]

    # 1) Type3：只有 image 字段，且 entries 数量 < available_images
    type3_json = base_dir / "type3.json"
    type3_json.write_text(
        json.dumps([{"image": "img1.jpg"}]), encoding="utf-8"
    )
    dataset_type3 = loader._load_mixed_from_jsonl(
        type3_json, base_dir, images, file_suffix=".json"
    )
    assert len(dataset_type3) == 1
    assert dataset_type3[0].image is not None

    # 2) Type4：image+text + text-only
    type4_json = base_dir / "type4.json"
    type4_json.write_text(
        json.dumps(
            [
                {"image": "img1.jpg", "text": "img sample"},
                {"text": "only text"},
            ]
        ),
        encoding="utf-8",
    )
    dataset_type4 = loader._load_mixed_from_jsonl(
        type4_json, base_dir, images, file_suffix=".json"
    )
    assert len(dataset_type4) == 2
    assert any(s.image is None for s in dataset_type4)
    assert any(s.image is not None for s in dataset_type4)

    # 3) 无有效条目：仅包含无效结构
    invalid_json = base_dir / "invalid_items.json"
    invalid_json.write_text(
        json.dumps([{"foo": "bar"}]), encoding="utf-8"
    )
    with pytest.raises(InvalidDatasetError) as exc_info:
        loader._load_mixed_from_jsonl(
            invalid_json, base_dir, images, file_suffix=".json"
        )
    assert "No valid entries found in JSONL file" in str(exc_info.value)


def test_parse_jsonl_line_all_branches(tmp_path: Path):
    """
    覆盖 _parse_jsonl_line 的主要分支：
    - 纯文本行
    - image+text
    - 仅 text
    - 缺少字段 / 非法类型
    """
    # 构造一个最小 loader 与环境
    loader = VLMDatasetLoader()
    base_dir = tmp_path
    img = tmp_path / "img.jpg"
    img.write_bytes(b"1")
    image_map = {img.name: img}

    # 纯字符串
    entry, is_img, is_txt = loader._parse_jsonl_line(
        json.dumps("hello"), 1, base_dir, image_map
    )
    assert isinstance(entry, VlmCalibSample)
    assert is_img == 0 and is_txt == 1

    # image + text
    payload = {"image": "img.jpg", "text": "pic"}
    entry, is_img, is_txt = loader._parse_jsonl_line(
        json.dumps(payload), 2, base_dir, image_map
    )
    assert isinstance(entry, VlmCalibSample)
    assert is_img == 1 and is_txt == 0
    assert Path(entry.image).name == "img.jpg"

    # 仅 text 字段
    payload = {"text": "only text"}
    entry, is_img, is_txt = loader._parse_jsonl_line(
        json.dumps(payload), 3, base_dir, image_map
    )
    assert isinstance(entry, VlmCalibSample)
    assert is_img == 0 and is_txt == 1

    # 缺少必要字段 -> 返回 None
    payload = {"invalid": "field"}
    entry, is_img, is_txt = loader._parse_jsonl_line(
        json.dumps(payload), 4, base_dir, image_map
    )
    assert entry is None and is_img == 0 and is_txt == 0


def test_parse_jsonl_line_json_error_branches(tmp_path: Path):
    """
    覆盖 `_parse_jsonl_line` 中 JSONDecodeError 和通用异常分支（507-512）。
    """
    loader = VLMDatasetLoader()
    base_dir = tmp_path
    image_map = {}

    # 1) JSONDecodeError：传入非法 JSON 字符串
    entry, is_img, is_txt = loader._parse_jsonl_line(
        "{ bad json", 1, base_dir, image_map
    )
    assert entry is None and is_img == 0 and is_txt == 0

    # 2) 通用异常：通过 patch json.loads 抛出 RuntimeError
    with patch(
        "msmodelslim.infra.vlm_dataset_loader.json.loads",
        side_effect=RuntimeError("generic error"),
    ):
        entry2, is_img2, is_txt2 = loader._parse_jsonl_line(
            '"ok string"', 2, base_dir, image_map
        )
        assert entry2 is None and is_img2 == 0 and is_txt2 == 0


def test_resolve_image_path_missing_relative_and_absolute_and_get_valid_error(tmp_path: Path):
    """
    覆盖 `_resolve_image_path` 中：
    - 相对路径找不到图片（561-564）。
    - 绝对路径图片不存在（566-569）。
    - `get_valid_read_path` 抛出异常被捕获（571-575）。
    """
    loader = VLMDatasetLoader()
    base_dir = tmp_path
    image_map = {}

    # 1) 相对路径缺失
    result = loader._resolve_image_path(
        "missing_rel.jpg", base_dir, image_map, hint="rel"
    )
    assert result is None

    # 2) 绝对路径缺失
    abs_missing = (tmp_path / "abs_missing.jpg").resolve()
    result2 = loader._resolve_image_path(
        str(abs_missing), base_dir, image_map, hint="abs"
    )
    assert result2 is None

    # 3) get_valid_read_path 抛出异常
    ok_img = tmp_path / "ok.jpg"
    ok_img.write_bytes(b"1")
    image_map2 = {ok_img.name: ok_img}

    with patch(
        "msmodelslim.infra.vlm_dataset_loader.get_valid_read_path",
        side_effect=RuntimeError("validate fail"),
    ):
        result3 = loader._resolve_image_path(
            ok_img.name, base_dir, image_map2, hint="gvp"
        )
        assert result3 is None
