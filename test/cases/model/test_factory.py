# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
from pathlib import Path
from unittest.mock import Mock, patch
from importlib.metadata import EntryPoints, EntryPoint
import pytest

from msmodelslim.model.plugin_factory import PluginModelFactory, DEFAULT
from msmodelslim.utils.exception import UnsupportedError


class DummyAdapter:
    def __init__(self, model_type, model_path, trust_remote_code):
        self.model_type = model_type
        self.model_path = model_path
        self.trust_remote_code = trust_remote_code


def make_entry_point(name):
    ep = Mock(spec=EntryPoint)
    ep.name = name
    ep.load.return_value = DummyAdapter
    return ep


@patch("msmodelslim.model.plugin_factory.DependencyChecker.check_plugin")
@patch("msmodelslim.model.plugin_factory.entry_points")
def test_create_valid_model(mock_entry_points, mock_check_plugin):
    mock_check_plugin.return_value = None
    PluginModelFactory._model_map = None
    ep = make_entry_point("deepseek")
    eps = EntryPoints([ep])
    mock_entry_points.return_value.select.return_value = eps

    model = PluginModelFactory().create("deepseek", Path("/tmp/path"))

    ep.load.assert_called_once()
    assert isinstance(model, DummyAdapter)
    assert model.model_type == "deepseek"


@patch("msmodelslim.model.plugin_factory.entry_points")
@patch("msmodelslim.model.plugin_factory.get_logger")
@patch("msmodelslim.model.plugin_factory.DependencyChecker.check_plugin")
def test_create_fallback_default(mock_check_plugin, mock_logger, mock_entry_points):
    # Only default exists
    PluginModelFactory._model_map = None
    ep_default = make_entry_point(DEFAULT)
    eps = EntryPoints([ep_default])
    mock_entry_points.return_value.select.return_value = eps
    mock_check_plugin.return_value = None

    model = PluginModelFactory().create("not_exist", Path("/tmp/path"))

    mock_logger().warning.assert_called_once()
    assert model.model_type == DEFAULT


@patch("msmodelslim.model.plugin_factory.entry_points")
@patch("msmodelslim.model.plugin_factory.DependencyChecker.check_plugin")
def test_no_adapter_registered_should_raise(mock_check_plugin, mock_entry_points):
    # No adapters at all
    PluginModelFactory._model_map = None
    eps = EntryPoints([])
    mock_entry_points.return_value.select.return_value = eps
    mock_check_plugin.return_value = None

    with pytest.raises(UnsupportedError):
        PluginModelFactory().create("not_exist", Path("/tmp/path"))
