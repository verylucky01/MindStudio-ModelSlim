from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from msmodelslim.core.const import DeviceType
from msmodelslim.model.kimi_k2_5 import model_adapter as target
from msmodelslim.model.kimi_k2_5.model_adapter import KimiK25ModelAdapter, default_dtype
from msmodelslim.utils.exception import UnsupportedError


def _adapter(**kwargs):
    a = KimiK25ModelAdapter.__new__(KimiK25ModelAdapter)
    for k, v in kwargs.items():
        setattr(a, k, v)
    return a


def _forward_model(mm_projector=object(), vision_dtype=torch.float32):
    class M:
        def __init__(self):
            self.language_model = SimpleNamespace(model=SimpleNamespace(embed_tokens=nn.Embedding(10, 4)))
            self.vision_tower = SimpleNamespace(
                patch_embed=SimpleNamespace(proj=SimpleNamespace(weight=torch.ones(1, dtype=vision_dtype)))
            )
            self.mm_projector = mm_projector

        def _merge_input_ids_with_image_features(self, **kwargs):
            ids = kwargs["input_ids"]
            return kwargs["inputs_embeds"], kwargs["attention_mask"], None, torch.arange(ids.shape[1]).unsqueeze(0)

    return M()


def _init_fake_model(mm_projector=object(), with_heads=True):
    text_config = SimpleNamespace(num_attention_heads=16, num_key_value_heads=8) if with_heads else SimpleNamespace()

    class FakeModel:
        def __init__(self):
            self.vision_tower = object()
            self.mm_projector = mm_projector
            self.language_model = SimpleNamespace(lm_head=object())
            self.config = SimpleNamespace(text_config=text_config)

        def load_state_dict(self, _state_dict, **_kwargs):
            return None

        def eval(self):
            return None

    return FakeModel()


class _FakeSafeOpen:
    def __init__(self, collector=None):
        self.collector = collector

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def get_tensor(self, name):
        if self.collector is not None:
            self.collector.append(name)
        return torch.ones((2, 2), dtype=torch.float32) if name.endswith("weight") else torch.ones(2, dtype=torch.float32)


@pytest.mark.parametrize(
    "fn, expected",
    [
        (lambda a: a.get_model_pedigree(), "kimi_k2_5"),
        (lambda a: a.get_layer_wise_offload_device(), "meta"),
    ],
)
def test_basic_getters(fn, expected):
    assert fn(_adapter()) == expected


def test_get_model_type_given_model_type_when_called_then_return_model_type():
    assert _adapter(model_type="kimi").get_model_type() == "kimi"


def test_default_dtype_given_dtype_when_context_exit_then_restore_original():
    original = torch.get_default_dtype()
    with default_dtype(torch.bfloat16):
        assert torch.get_default_dtype() == torch.bfloat16
    assert torch.get_default_dtype() == original


@pytest.mark.parametrize(
    "sample",
    [SimpleNamespace(image=None, text="hi"), SimpleNamespace(image="a.jpg", text=None)],
)
def test_handle_dataset_given_item_missing_modality_then_raise_unsupported_error(monkeypatch, sample):
    adapter = _adapter(model_path=Path("."), trust_remote_code=False)
    adapter._collect_inputs_to_device = lambda *args, **kwargs: {}

    class DummyProcessor:
        def __call__(self, messages=None, return_tensors="pt"):
            return {"input_ids": torch.ones((1, 2), dtype=torch.long)}

    monkeypatch.setattr(target.AutoProcessor, "from_pretrained", lambda *args, **kwargs: DummyProcessor())
    with pytest.raises(UnsupportedError):
        adapter.handle_dataset([sample], DeviceType.CPU)


@pytest.mark.parametrize(
    "tokenizer_factory",
    [
        lambda: (_ for _ in ()).throw(RuntimeError("t")),
        lambda: object(),
    ],
)
def test_handle_dataset_given_processor_fails_then_raise_unsupported_error(monkeypatch, tokenizer_factory):
    adapter = _adapter(model_path=Path("."), trust_remote_code=False)
    monkeypatch.setattr(target.AutoProcessor, "from_pretrained", lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("p")))
    monkeypatch.setattr(target.AutoTokenizer, "from_pretrained", lambda *args, **kwargs: tokenizer_factory())
    with pytest.raises(UnsupportedError):
        adapter.handle_dataset([], DeviceType.CPU)


def test_handle_dataset_given_valid_item_when_called_then_return_processed_data(monkeypatch):
    adapter = _adapter(model_path=Path("."), trust_remote_code=False)

    class DummyProcessor:
        def __call__(self, messages=None, return_tensors="pt"):
            return {
                "input_ids": torch.ones((1, 2), dtype=torch.long),
                "pixel_values": torch.ones((1, 3, 2, 2), dtype=torch.float32),
                "grid_thws": torch.ones((1, 3), dtype=torch.long),
                "attention_mask": torch.ones((1, 2), dtype=torch.long),
            }

    monkeypatch.setattr(target.AutoProcessor, "from_pretrained", lambda *args, **kwargs: DummyProcessor())
    monkeypatch.setattr(target, "get_valid_read_path", lambda p, **kwargs: p)
    adapter._collect_inputs_to_device = lambda inputs, device, keys, defaults: {k: inputs[k] for k in keys if k in inputs}
    out = adapter.handle_dataset([SimpleNamespace(image="a.jpg", text="hello")], DeviceType.CPU)
    assert isinstance(out, list)
    assert len(out) == 1


def test_generate_decoder_layer_given_num_layers_when_called_then_return_all_layers():
    adapter = _adapter(config=SimpleNamespace(text_config=SimpleNamespace(num_hidden_layers=2)))
    adapter._load_decoder_if_not_exist = lambda model, name, idx: f"layer_{idx}"
    assert list(adapter.generate_decoder_layer(object())) == [
        ("language_model.model.layers.0", "layer_0"),
        ("language_model.model.layers.1", "layer_1"),
    ]


def test_enable_kv_cache_given_need_flag_when_called_then_set_use_cache():
    model = SimpleNamespace(config=SimpleNamespace(use_cache=False))
    _adapter().enable_kv_cache(model, True)
    assert model.config.use_cache is True


def test_get_adapter_config_for_subgraph_given_layers_when_called_then_return_configs():
    adapter = _adapter(config=SimpleNamespace(text_config=SimpleNamespace(
        num_hidden_layers=3,
        num_attention_heads=8,
        num_key_value_heads=4,
        qk_nope_head_dim=128,
        v_head_dim=128,
    )))

    out = adapter.get_adapter_config_for_subgraph()
    ov = [cfg for cfg in out if cfg.subgraph_type == "ov"]
    norm = [cfg for cfg in out if cfg.subgraph_type == "norm-linear"]

    assert len(out) == 6
    assert len(ov) == 2
    assert len(norm) == 4
    assert ov[0].mapping.source == "language_model.model.layers.0.self_attn.kv_b_proj"
    assert ov[0].mapping.targets == ["language_model.model.layers.0.self_attn.o_proj"]
    assert ov[0].fusion.num_attention_heads == 8
    assert ov[0].fusion.num_key_value_heads == 4
    assert ov[0].fusion.custom_config["qk_nope_head_dim"] == 128
    assert ov[0].fusion.custom_config["v_head_dim"] == 128


def test_ascendv1_save_postprocess_given_tiktoken_exists_when_called_then_copy_and_chmod(monkeypatch, tmp_path):
    adapter = _adapter(model_path=str(tmp_path))
    (tmp_path / "tiktoken.model").write_text("x", encoding="utf-8")

    called = {"copy": 0, "chmod": 0}
    monkeypatch.setattr(target, "safe_copy_file", lambda src_path, dest_path: called.__setitem__("copy", 1))
    monkeypatch.setattr(target.os, "chmod", lambda p, m: called.__setitem__("chmod", 1))

    adapter.ascendv1_save_postprocess(nn.Linear(2, 2), str(tmp_path / "save"))
    assert called["copy"] == 1
    assert called["chmod"] == 1


def test_get_weight_map_given_index_json_when_loaded_then_return_weight_map(monkeypatch):
    adapter = _adapter(model_path="/tmp/model")
    monkeypatch.setattr(target, "json_safe_load", lambda p: {"weight_map": {"a": "f"}})
    assert adapter._get_weight_map() == {"a": "f"}


@pytest.mark.parametrize(
    "weight_map,prefix,expected_names",
    [
        ({"weight": "a.safetensors", "bias": "a.safetensors"}, "", None),
        (
            {
                "language_model.model.layers.0.weight": "a.safetensors",
                "language_model.model.layers.0.bias": "a.safetensors",
            },
            "language_model.model.layers.0",
            {"language_model.model.layers.0.weight", "language_model.model.layers.0.bias"},
        ),
    ],
)
def test_get_state_dict_paths(monkeypatch, weight_map, prefix, expected_names):
    adapter = _adapter(model_path="/tmp/model")
    adapter._get_weight_map = lambda: weight_map
    module = nn.Linear(2, 2)
    monkeypatch.setattr(target, "get_valid_read_path", lambda p, **kwargs: p)

    called = []
    monkeypatch.setattr(target, "safe_open", lambda *args, **kwargs: _FakeSafeOpen(called if expected_names else None))

    out = adapter._get_state_dict(module, prefix=prefix)
    assert "weight" in out and "bias" in out
    if expected_names is not None:
        assert expected_names.issubset(set(called))


def test_load_decoder_if_not_exist_given_loaded_decoder_when_access_ok_then_return_loaded():
    adapter = _adapter()

    class L(nn.Module):
        def __init__(self):
            super().__init__()
            self.input_layernorm = SimpleNamespace(weight=torch.ones(1))

    layer = L()

    class M:
        def get_submodule(self, _name):
            return layer

    assert adapter._load_decoder_if_not_exist(M(), "language_model.model.layers.0", 0) is layer


def test_load_decoder_if_not_exist_given_missing_layer_cls_when_called_then_raise_unsupported_error():
    adapter = _adapter(config=SimpleNamespace(text_config=SimpleNamespace()), model_path="/tmp/model")

    class M:
        def get_submodule(self, _name):
            raise AttributeError("missing")

        language_model = SimpleNamespace(model=SimpleNamespace(layers=[]))

    with pytest.raises(UnsupportedError):
        adapter._load_decoder_if_not_exist(M(), "language_model.model.layers.1", 1)


@pytest.mark.parametrize(
    "inputs, mm_projector, vision_dtype, expected",
    [
        ({"input_ids": torch.tensor([[1, 2]], dtype=torch.long), "pixel_values": None}, object(), torch.float32, ["language_model.model.layers.0"]),
        (
            {
                "input_ids": torch.tensor([[1, 2]], dtype=torch.long),
                "attention_mask": torch.ones((1, 2), dtype=torch.long),
                "pixel_values": torch.ones((1, 3, 2, 2), dtype=torch.float32),
                "grid_thws": torch.ones((1, 3), dtype=torch.long),
            },
            object(),
            torch.float32,
            ["vision_tower", "mm_projector", "language_model.model.layers.0"],
        ),
        (
            [{
                "input_ids": torch.tensor([[1, 2]], dtype=torch.long),
                "attention_mask": None,
                "pixel_values": torch.ones((1, 3, 2, 2), dtype=torch.float32),
                "grid_thws": None,
            }],
            None,
            torch.float16,
            ["vision_tower", "language_model.model.layers.0"],
        ),
        (
            {"input_ids": torch.tensor([[1]], dtype=torch.long), "pixel_values": torch.ones((1, 3, 2, 2), dtype=torch.float32)},
            object(),
            torch.float32,
            ["language_model.model.layers.0"],
        ),
        (
            {"input_ids": torch.tensor([[1, 2]], dtype=torch.long), "pixel_values": torch.empty((0, 3, 2, 2), dtype=torch.float32)},
            object(),
            torch.float32,
            ["language_model.model.layers.0"],
        ),
    ],
)
def test_generate_model_forward_paths(monkeypatch, inputs, mm_projector, vision_dtype, expected):
    adapter = _adapter()

    class EchoLayer(nn.Module):
        def forward(self, hidden_states, **kwargs):
            return hidden_states

    adapter.generate_decoder_layer = lambda model: iter([("language_model.model.layers.0", EchoLayer())])
    monkeypatch.setattr(target, "_prepare_4d_causal_attention_mask", lambda *args, **kwargs: torch.ones((1, 1, 2, 2)))

    gen = adapter.generate_model_forward(_forward_model(mm_projector, vision_dtype), inputs)

    req = next(gen)
    got = [req.name]
    if req.name == "vision_tower":
        if len(expected) == 3:
            req = gen.send([torch.ones((1, 2, 4), dtype=torch.float32)])
            got.append(req.name)
            req = gen.send([torch.ones((1, 2, 4), dtype=torch.float32)])
            got.append(req.name)
        else:
            req = gen.send([torch.ones((1, 2, 4), dtype=torch.float16)])
            got.append(req.name)
    assert got == expected


def test_generate_model_forward_given_decoder_returns_tuple_when_called_then_unwrap_hidden_states(monkeypatch):
    adapter = _adapter()

    class TupleLayer(nn.Module):
        def forward(self, hidden_states, **kwargs):
            return (hidden_states + 1,)

    adapter.generate_decoder_layer = lambda model: iter([
        ("language_model.model.layers.0", TupleLayer()),
        ("language_model.model.layers.1", nn.Identity()),
    ])

    model = SimpleNamespace(
        language_model=SimpleNamespace(model=SimpleNamespace(embed_tokens=nn.Embedding(10, 4))),
        vision_tower=SimpleNamespace(patch_embed=SimpleNamespace(proj=SimpleNamespace(weight=torch.ones(1, dtype=torch.float32)))),
    )
    monkeypatch.setattr(target, "_prepare_4d_causal_attention_mask", lambda *args, **kwargs: torch.ones((1, 1, 2, 2)))

    gen = adapter.generate_model_forward(model, {"input_ids": torch.tensor([[1, 2]], dtype=torch.long), "pixel_values": None})
    assert next(gen).name == "language_model.model.layers.0"
    assert gen.send((torch.ones((1, 2, 4), dtype=torch.float32),)).name == "language_model.model.layers.1"


def test_load_decoder_if_not_exist_given_meta_like_decoder_when_called_then_replace_in_module_list(monkeypatch):
    adapter = _adapter(config=SimpleNamespace(text_config=SimpleNamespace()), model_path="/tmp/model")

    class DummyLayer(nn.Module):
        def __init__(self, config=None, layer_idx=0):
            super().__init__()
            self.input_layernorm = nn.LayerNorm(1)

    class MetaWeight:
        @property
        def device(self):
            raise RuntimeError("meta")

    loaded_decoder = SimpleNamespace(input_layernorm=SimpleNamespace(weight=MetaWeight()))
    module_list = [DummyLayer()]

    class M:
        def get_submodule(self, _name):
            return loaded_decoder

        language_model = SimpleNamespace(model=SimpleNamespace(layers=module_list))

    monkeypatch.setattr(adapter, "_get_state_dict", lambda decoder, prefix="": {})
    monkeypatch.setattr(target, "auto_convert_module_int4_to_bf16", lambda *args, **kwargs: None)

    out = adapter._load_decoder_if_not_exist(M(), "language_model.model.layers.0", 0)
    assert isinstance(out, DummyLayer)
    assert isinstance(module_list[0], DummyLayer)


def test_generate_model_visit_given_model_when_called_then_yield_vision_mm_and_decoder(monkeypatch):
    adapter = _adapter()
    adapter.generate_decoder_layer = lambda model: iter([("language_model.model.layers.0", nn.Identity())])

    def fake_generated_decoder_layer_visit_func(_model, transformer_blocks):
        for name, layer in transformer_blocks:
            yield target.ProcessRequest(name=name, module=layer, args=(), kwargs={})

    monkeypatch.setattr(target, "generated_decoder_layer_visit_func", fake_generated_decoder_layer_visit_func)
    requests = list(adapter.generate_model_visit(SimpleNamespace(vision_tower=object(), mm_projector=object())))

    assert requests[0].name == "vision_tower"
    assert requests[1].name == "mm_projector"
    assert requests[2].name == "language_model.model.layers.0"


@pytest.mark.parametrize(
    "mm_projector, config, with_heads, expected_replace_calls, expected_layers",
    [
        (
            object(),
            SimpleNamespace(
                text_config=SimpleNamespace(num_hidden_layers=4, num_attention_heads=16, num_key_value_heads=8),
                vision_config=SimpleNamespace(vt_num_hidden_layers=2),
                use_cache=True,
            ),
            True,
            None,
            4,
        ),
        (
            None,
            SimpleNamespace(
                text_config=SimpleNamespace(num_hidden_layers=2),
                vision_config=SimpleNamespace(vt_num_hidden_layers=1),
                use_cache=True,
            ),
            False,
            2,
            2,
        ),
    ],
)
def test_init_model_main_paths(monkeypatch, mm_projector, config, with_heads, expected_replace_calls, expected_layers):
    adapter = _adapter(model_path="/tmp/model", trust_remote_code=False, config=config)
    fake_model = _init_fake_model(mm_projector=mm_projector, with_heads=with_heads)

    monkeypatch.setattr(target, "get_valid_read_path", lambda p, **kwargs: p)
    monkeypatch.setattr(target.SafeGenerator, "get_model_from_pretrained", lambda **kwargs: fake_model)
    monkeypatch.setattr(adapter, "_get_state_dict", lambda model: {})
    monkeypatch.setattr(target, "auto_convert_module_int4_to_bf16", lambda *args, **kwargs: None)

    replace_calls = {"count": 0}

    def fake_replace(module, prefix, _model_path):
        replace_calls["count"] += 1
        return (prefix, module)

    monkeypatch.setattr(target, "replace_compressed_linear_with_bf16", fake_replace)

    out = adapter.init_model(DeviceType.CPU)
    assert out is fake_model
    assert adapter.config.text_config.num_hidden_layers == expected_layers
    assert adapter.config.use_cache is False

    if with_heads:
        assert fake_model.config.num_attention_heads == 16
        assert fake_model.config.num_key_value_heads == 8

    if expected_replace_calls is not None:
        assert replace_calls["count"] == expected_replace_calls
    else:
        assert fake_model.vision_tower[0] == "vision_tower"
        assert fake_model.mm_projector[0] == "mm_projector"
        assert fake_model.language_model.lm_head[0] == "language_model.lm_head"


def test_init_model_given_load_model_raises_when_called_then_restore_initialize_missing_keys(monkeypatch):
    adapter = _adapter(
        model_path="/tmp/model",
        trust_remote_code=False,
        config=SimpleNamespace(
            text_config=SimpleNamespace(num_hidden_layers=4),
            vision_config=SimpleNamespace(vt_num_hidden_layers=2),
            use_cache=True,
        ),
    )

    monkeypatch.setattr(target, "get_valid_read_path", lambda p, **kwargs: p)
    original = target.PreTrainedModel._initialize_missing_keys

    monkeypatch.setattr(target.SafeGenerator, "get_model_from_pretrained", lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("boom")))
    with pytest.raises(RuntimeError):
        adapter.init_model(DeviceType.CPU)

    assert target.PreTrainedModel._initialize_missing_keys is original
    # 当前实现在异常路径不会恢复 num_hidden_layers，这里仅看护 monkey patch 回滚
    assert adapter.config.text_config.num_hidden_layers == 1
