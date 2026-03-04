#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
-------------------------------------------------------------------------
MindStudio - Mulan PSL v2. See http://license.coscl.org.cn/MulanPSL2
-------------------------------------------------------------------------
"""

import pytest

from msmodelslim.core.context.shared_dict_context.context import SharedDictContext


class CustomInvalidType:
    def __init__(self, name: str = "invalid") -> None:
        self.name = name

def _worker_read_ns(shared_ctx, ev, ns_key, expected, tensor_keys):
    ev.wait(timeout=10)
    ns = shared_ctx[ns_key]
    for k, exp in expected.items():
        assert (__import__("torch").allclose(ns.debug.get(k), exp) if k in tensor_keys else ns.debug.get(k) == exp)
        assert (__import__("torch").allclose(ns.state.get(f"state_{k}"), exp) if k in tensor_keys else ns.state.get(f"state_{k}") == exp)


def _worker_invalid_type(shared_ctx, ev, ns_key, attr, queue):
    ev.wait(timeout=10)
    ns = shared_ctx[ns_key]
    try:
        getattr(ns, attr)["x"] = CustomInvalidType(attr)
        queue.put((attr, False, ""))
    except Exception as e:
        queue.put((attr, "Unsupported value type" in str(e), str(e)))


def test_multi_process_rejects_non_whitelist_type():
    """State and debug reject custom type in child process."""
    mp = pytest.importorskip("torch.multiprocessing")
    ctx = mp.get_context("spawn") if hasattr(mp, "get_context") else mp
    parent_ctx = SharedDictContext(enable_debug=True)
    ns = parent_ctx["invalid_ns"]
    ev, q = ctx.Event(), ctx.Queue()
    p1 = ctx.Process(target=_worker_invalid_type, args=(parent_ctx, ev, "invalid_ns", "state", q))
    p2 = ctx.Process(target=_worker_invalid_type, args=(parent_ctx, ev, "invalid_ns", "debug", q))
    p1.start(); p2.start(); ev.set(); p1.join(timeout=30); p2.join(timeout=30)
    assert p1.exitcode == 0 and p2.exitcode == 0
    res = {}
    for _ in range(2):
        attr, rejected, msg = q.get(timeout=5)
        res[attr] = (attr, rejected, msg)
    for attr in ("state", "debug"):
        assert res[attr][1] is True, res[attr][2]


def test_shared_namespace_all_supported_types():
    """Basic types, collections, tensors, nested structures in shared namespace."""
    torch = pytest.importorskip("torch")
    ctx = SharedDictContext(enable_debug=True)
    ns = ctx["test"]
    ns.state["string"] = "hello"
    ns.state["int"] = 42
    ns.state["float"] = 3.14
    ns.state["bool"] = True
    ns.state["bytes"] = b"data"
    ns.state["none"] = None
    ns.state["dict"] = {"a": 1, "c": [1, 2, 3]}
    ns.state["list"] = [1, "two", 3.0, True]
    ns.state["tuple"] = (1, 2, 3)
    t = torch.tensor([1, 2, 3], device="cpu")
    ns.state["tensor"] = t
    nested = {"config": {"lr": 0.001, "layers": [64, 128]}, "weights": torch.randn(10, 10, device="cpu")}
    ns.state["nested"] = nested
    assert ns.state["string"] == "hello" and ns.state["int"] == 42 and ns.state["float"] == 3.14
    assert ns.state["bool"] is True and ns.state["bytes"] == b"data" and ns.state["none"] is None
    assert ns.state["dict"] == {"a": 1, "c": [1, 2, 3]} and ns.state["list"] == [1, "two", 3.0, True]
    assert ns.state["tuple"] == (1, 2, 3) and torch.allclose(ns.state["tensor"], t)
    r = ns.state["nested"]
    assert r["config"]["lr"] == 0.001 and torch.allclose(r["weights"], nested["weights"])
