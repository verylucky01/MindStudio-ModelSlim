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
Test cases for DistributedAscendV1Saver.
"""

import json
import os
import shutil
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock

import pytest
import torch
from torch import nn

from msmodelslim.app.quant_service.modelslim_v1.save.ascendv1 import AscendV1Config
from msmodelslim.app.quant_service.modelslim_v1.save.ascendv1_distributed import (
    DistributedAscendV1Config,
    DistributedAscendV1Saver,
    convert_to_distributed_config_if_needed,
    save_this_rank_only,
    decorate_on_methods,
)
from msmodelslim.core.base.protocol import BatchProcessRequest


class SimpleModel(nn.Module):
    """Simple model for testing."""
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(10, 20)
        self.linear2 = nn.Linear(20, 10)
    
    def forward(self, x):
        x = self.linear1(x)
        x = self.linear2(x)
        return x


class MockAdapter:
    """Mock adapter for testing."""
    def __init__(self, model_path):
        self._model_path = Path(model_path)
    
    @property
    def model_path(self):
        return self._model_path


@pytest.fixture
def temp_dir():
    """Create a temporary directory for testing."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)


@pytest.fixture
def mock_model():
    """Create a mock model for testing."""
    return SimpleModel()


@pytest.fixture
def mock_adapter(temp_dir):
    """Create a mock adapter for testing."""
    # Create a source directory with some config files
    source_dir = os.path.join(temp_dir, "source")
    os.makedirs(source_dir, exist_ok=True)
    
    # Create config.json
    config_path = os.path.join(source_dir, "config.json")
    with open(config_path, "w") as f:
        json.dump({"model_type": "test"}, f)
    
    return MockAdapter(source_dir)


@pytest.fixture
def mock_adapter_with_interface(temp_dir):
    """Create a mock adapter that implements AscendV1SaveInterface."""
    source_dir = os.path.join(temp_dir, "source")
    os.makedirs(source_dir, exist_ok=True)
    
    config_path = os.path.join(source_dir, "config.json")
    with open(config_path, "w") as f:
        json.dump({"model_type": "test"}, f)
    
    adapter = MagicMock()
    adapter.model_path = Path(source_dir)
    return adapter


@pytest.fixture
def setup_rank_directories(temp_dir):
    """Set up rank directories with test files."""
    for rank in range(2):
        rank_dir = os.path.join(temp_dir, f"rank_{rank}")
        os.makedirs(rank_dir, exist_ok=True)
        
        # Create safetensors file
        safetensors_file = os.path.join(rank_dir, f"quant_model_weights-{rank+1:05d}-of-00002.safetensors")
        with open(safetensors_file, "wb") as f:
            f.write(b"dummy_content")
        
        # Create index file
        index_file = os.path.join(rank_dir, "quant_model_weights.safetensors.index.json")
        index_data = {
            "metadata": {"total_size": 100},
            "weight_map": {
                f"layer_{rank}.weight": f"quant_model_weights-{rank+1:05d}-of-00002.safetensors"
            }
        }
        with open(index_file, "w") as f:
            json.dump(index_data, f)
        
        # Create description json file
        desc_file = os.path.join(rank_dir, "quant_model_description.json")
        desc_data = {f"layer_{rank}.weight": "W8A8"}
        with open(desc_file, "w") as f:
            json.dump(desc_data, f)
    
    return temp_dir


class TestDistributedAscendV1Config:
    """Test cases for DistributedAscendV1Config."""
    
    @staticmethod
    def test_config_basic():
        """Test config basic properties and inheritance."""
        config = DistributedAscendV1Config(
            save_directory="/test/path",
            part_file_size=8,
            ext={"key": "value"}
        )
        assert config.type == "ascendv1_saver_distributed"
        assert isinstance(config, AscendV1Config)
        assert config.save_directory == "/test/path"
        assert config.part_file_size == 8
        assert config.ext == {"key": "value"}


class TestConvertToDistributedConfigIfNeeded:
    """Test cases for convert_to_distributed_config_if_needed function."""
    
    @patch('msmodelslim.app.quant_service.modelslim_v1.save.ascendv1_distributed.dist.is_initialized')
    def test_returns_original_when_dist_not_initialized(self, mock_is_init):
        """Test that original configs are returned when dist is not initialized."""
        mock_is_init.return_value = False
        configs = [AscendV1Config(save_directory="/test")]
        result = convert_to_distributed_config_if_needed(configs)
        assert result == configs
        assert isinstance(result[0], AscendV1Config)
        assert not isinstance(result[0], DistributedAscendV1Config)
    
    @patch('torch.distributed.get_rank')
    @patch('torch.distributed.is_initialized')
    @patch('msmodelslim.app.quant_service.modelslim_v1.save.ascendv1_distributed.dist.is_initialized')
    def test_converts_ascendv1_config_to_distributed(self, mock_is_init, mock_global_is_init, mock_global_get_rank):
        """Test that AscendV1Config is converted to DistributedAscendV1Config."""
        mock_is_init.return_value = True
        mock_global_is_init.return_value = True
        mock_global_get_rank.return_value = 0
        configs = [AscendV1Config(save_directory="/test", part_file_size=4)]
        result = convert_to_distributed_config_if_needed(configs)
        
        assert len(result) == 1
        assert isinstance(result[0], DistributedAscendV1Config)
        assert result[0].save_directory == "/test"
        assert result[0].part_file_size == 4


class TestSaveThisRankOnlyDecorator:
    """Test cases for save_this_rank_only decorator."""
    
    @patch('msmodelslim.app.quant_service.modelslim_v1.save.ascendv1_distributed.dist.is_initialized')
    def test_decorator_calls_func_when_dist_not_initialized(self, mock_is_init):
        """Test that function is called when dist is not initialized."""
        mock_is_init.return_value = False
        
        call_count = [0]
        
        @save_this_rank_only()
        def test_func(self, prefix, module):
            call_count[0] += 1
        
        mock_instance = MagicMock()
        test_func(mock_instance, "prefix", nn.Linear(10, 10))
        assert call_count[0] == 1
    
    @patch('msmodelslim.app.quant_service.modelslim_v1.save.ascendv1_distributed.dist.is_initialized')
    def test_decorator_with_dist_helper(self, mock_is_init):
        """Test decorator behavior with dist_helper."""
        mock_is_init.return_value = True
        
        call_count = [0]
        
        @save_this_rank_only()
        def test_func(self, prefix, module):
            call_count[0] += 1
        
        mock_instance = MagicMock()
        mock_instance.dist_helper = MagicMock()
        
        # Test local-only module calls function
        mock_instance.dist_helper.is_local_only.return_value = True
        mock_instance.shared_modules_slice = []
        call_count[0] = 0
        test_func(mock_instance, "prefix", nn.Linear(10, 10))
        assert call_count[0] == 1
        
        # Test shared module in slice calls function
        mock_instance.dist_helper.is_local_only.return_value = False
        mock_instance.shared_modules_slice = ["my_prefix"]
        call_count[0] = 0
        test_func(mock_instance, "my_prefix", nn.Linear(10, 10))
        assert call_count[0] == 1
        
        # Test module not in slice skips function
        mock_instance.shared_modules_slice = ["other_prefix"]
        call_count[0] = 0
        test_func(mock_instance, "my_prefix", nn.Linear(10, 10))
        assert call_count[0] == 0


class TestDecorateOnMethods:
    """Test cases for decorate_on_methods class decorator."""
    
    @staticmethod
    def test_decorates_on_methods():
        """Test that on_ methods are decorated."""
        @decorate_on_methods
        class TestClass:
            def on_test(self, prefix, module):
                pass
            
            def other_method(self):
                pass
        
        # Check that on_test has been wrapped
        assert hasattr(TestClass.on_test, '__wrapped__')
        # other_method should not have __wrapped__ attribute
        assert not hasattr(TestClass.other_method, '__wrapped__')


class TestDistributedAscendV1Saver:
    """Test cases for DistributedAscendV1Saver class."""
    
    @pytest.fixture(autouse=True)
    @staticmethod
    def setup_common_mocks():
        """Set up common mocks for all tests in this class."""
        dist_path = (
            'msmodelslim.app.quant_service.modelslim_v1.save.'
            'ascendv1_distributed.dist'
        )
        with patch('torch.distributed.get_rank', return_value=0), \
             patch('torch.distributed.is_initialized', return_value=True), \
             patch(f'{dist_path}.get_world_size', return_value=2), \
             patch(f'{dist_path}.get_rank', return_value=0), \
             patch(f'{dist_path}.is_initialized', return_value=True):
            yield
    
    @staticmethod
    def create_saver(temp_dir, mock_model, mock_adapter, **config_kwargs):
        """Helper method to create a saver with default or custom config."""
        config = DistributedAscendV1Config(
            save_directory=temp_dir, **config_kwargs
        )
        return DistributedAscendV1Saver(mock_model, config, mock_adapter)
    
    @staticmethod
    def create_saver_with_rank_dir(
        temp_dir, mock_model, mock_adapter, **config_kwargs
    ):
        """Helper method to create a saver with rank directory set."""
        config = DistributedAscendV1Config(
            save_directory=temp_dir, **config_kwargs
        )
        saver = DistributedAscendV1Saver(mock_model, config, mock_adapter)
        saver.save_directory = os.path.join(temp_dir, "rank_0")
        return saver
    
    @staticmethod
    def setup_rank_directories(temp_dir):
        """Set up rank directories with test files."""
        for rank in range(2):
            rank_dir = os.path.join(temp_dir, f"rank_{rank}")
            os.makedirs(rank_dir, exist_ok=True)
            
            safetensors_file = os.path.join(
                rank_dir,
                f"quant_model_weights-{rank+1:05d}-of-00002.safetensors"
            )
            with open(safetensors_file, "wb") as f:
                f.write(b"dummy")
            
            desc_file = os.path.join(rank_dir, "quant_model_description.json")
            with open(desc_file, "w") as f:
                json.dump({f"layer_{rank}": "W8A8"}, f)
    
    @staticmethod
    def setup_writers(saver):
        """Set up default writers for saver."""
        saver.safetensors_writer = MagicMock()
        saver.safetensors_writer.save_prefix = "quant_model_weights"
        saver.json_writer = MagicMock()
        saver.json_writer.file_name = "quant_model_description.json"
    
    @staticmethod
    def test_cleanup_rank_dirs(
        setup_rank_directories, mock_model, mock_adapter
    ):
        """Test _cleanup_rank_dirs method."""
        temp_dir = setup_rank_directories
        
        saver = TestDistributedAscendV1Saver.create_saver_with_rank_dir(
            temp_dir, mock_model, mock_adapter
        )
        
        # Verify directories exist before cleanup
        assert os.path.exists(os.path.join(temp_dir, "rank_0"))
        assert os.path.exists(os.path.join(temp_dir, "rank_1"))
        
        saver._cleanup_rank_dirs()
        
        # Verify directories are removed after cleanup
        assert not os.path.exists(os.path.join(temp_dir, "rank_0"))
        assert not os.path.exists(os.path.join(temp_dir, "rank_1"))
    
    @staticmethod
    def test_get_rank_save_directory(temp_dir, mock_model, mock_adapter):
        """Test get_rank_save_directory method."""
        dist_path = (
            'msmodelslim.app.quant_service.modelslim_v1.save.'
            'ascendv1_distributed.dist'
        )
        # 修改特定参数：rank=1, 未初始化
        with patch('torch.distributed.get_rank', return_value=1), \
             patch('torch.distributed.is_initialized', return_value=False), \
             patch(f'{dist_path}.get_rank', return_value=1), \
             patch(f'{dist_path}.is_initialized', return_value=False):
            saver = TestDistributedAscendV1Saver.create_saver(
                temp_dir, mock_model, mock_adapter
            )
            
            result = saver.get_rank_save_directory()
            assert result == os.path.join(temp_dir, "rank_1")
    
    @staticmethod
    def test_init_with_distributed(temp_dir, mock_model, mock_adapter):
        """Test DistributedAscendV1Saver initialization."""
        saver = TestDistributedAscendV1Saver.create_saver(
            temp_dir, mock_model, mock_adapter
        )
        
        assert saver.save_directory == os.path.join(temp_dir, "rank_0")
        assert len(saver.file_mappings) == 2
        assert saver.dist_helper is None
    
    @staticmethod
    def test_merge_index_files(
        setup_rank_directories, mock_model, mock_adapter
    ):
        """Test _merge_index_files method."""
        temp_dir = setup_rank_directories
        
        saver = TestDistributedAscendV1Saver.create_saver_with_rank_dir(
            temp_dir, mock_model, mock_adapter
        )
        saver.safetensors_writer = MagicMock()
        saver.safetensors_writer.save_prefix = "quant_model_weights"
        
        # Set up file mappings
        saver.file_mappings = [
            {
                "quant_model_weights-00001-of-00002.safetensors":
                    "merged_00001.safetensors"
            },
            {
                "quant_model_weights-00002-of-00002.safetensors":
                    "merged_00002.safetensors"
            }
        ]
        
        saver._merge_index_files()
        
        # Check that merged index file was created
        merged_index_path = os.path.join(
            temp_dir, "quant_model_weights.safetensors.index.json"
        )
        assert os.path.exists(merged_index_path)
        
        with open(merged_index_path, "r") as f:
            merged_data = json.load(f)
        
        assert merged_data["metadata"]["total_size"] == 200  # 100 + 100
    
    @staticmethod
    def test_merge_json_files(
        setup_rank_directories, mock_model, mock_adapter
    ):
        """Test _merge_json_files method."""
        temp_dir = setup_rank_directories
        
        saver = TestDistributedAscendV1Saver.create_saver_with_rank_dir(
            temp_dir, mock_model, mock_adapter
        )
        saver.json_writer = MagicMock()
        saver.json_writer.file_name = "quant_model_description.json"
        
        saver._merge_json_files()
        
        # Check that merged json file was created
        merged_json_path = os.path.join(
            temp_dir, "quant_model_description.json"
        )
        assert os.path.exists(merged_json_path)
        
        with open(merged_json_path, "r") as f:
            merged_data = json.load(f)
        
        assert "layer_0.weight" in merged_data
        assert "layer_1.weight" in merged_data
    
    @staticmethod
    def test_merge_ranks_on_rank0(
        setup_rank_directories, mock_model, mock_adapter
    ):
        """Test merge_ranks method on rank 0."""
        temp_dir = setup_rank_directories
        dist_path = (
            'msmodelslim.app.quant_service.modelslim_v1.save.'
            'ascendv1_distributed.dist'
        )
        with patch(f'{dist_path}.barrier') as mock_barrier, \
             patch(f'{dist_path}.all_gather_object') as mock_all_gather:
            # Mock all_gather_object to set file counts
            def set_file_counts(output_list, local_count):
                output_list[0] = 1
                output_list[1] = 1
            mock_all_gather.side_effect = set_file_counts
            
            saver = TestDistributedAscendV1Saver.create_saver_with_rank_dir(
                temp_dir, mock_model, mock_adapter
            )
            TestDistributedAscendV1Saver.setup_writers(saver)
            
            saver.merge_ranks()
            
            mock_barrier.assert_called_once()
    
    @staticmethod
    def test_merge_safetensor_files(
        setup_rank_directories, mock_model, mock_adapter
    ):
        """Test _merge_safetensor_files method."""
        temp_dir = setup_rank_directories
        
        # 修改特定参数：part_file_size=4
        saver = TestDistributedAscendV1Saver.create_saver_with_rank_dir(
            temp_dir, mock_model, mock_adapter, part_file_size=4
        )
        saver.safetensors_writer = MagicMock()
        saver.safetensors_writer.save_prefix = "quant_model_weights"
        
        file_counts = [1, 1]
        saver._merge_safetensor_files(file_counts)
        
        # Check that file mappings were created
        assert len(saver.file_mappings[0]) == 1
        assert len(saver.file_mappings[1]) == 1
    
    @staticmethod
    def test_post_run_calls_merge_ranks_when_distributed(
        temp_dir, mock_model, mock_adapter_with_interface
    ):
        """Test that post_run calls merge_ranks when distributed."""
        dist_path = (
            'msmodelslim.app.quant_service.modelslim_v1.save.'
            'ascendv1_distributed'
        )
        with patch(f'{dist_path}.copy_files') as mock_copy_files, \
             patch(f'{dist_path}.remove_quantization_config'), \
             patch(f'{dist_path}.dist.barrier') as mock_barrier, \
             patch(f'{dist_path}.dist.all_gather_object') as mock_all_gather:
            # Set up rank directories
            TestDistributedAscendV1Saver.setup_rank_directories(temp_dir)
            
            def set_file_counts(output_list, local_count):
                output_list[0] = 1
                output_list[1] = 1
            mock_all_gather.side_effect = set_file_counts
            
            saver = TestDistributedAscendV1Saver.create_saver_with_rank_dir(
                temp_dir, mock_model, mock_adapter_with_interface
            )
            
            # Mock writers
            saver.json_writer = MagicMock()
            saver.json_writer.file_name = "quant_model_description.json"
            saver.safetensors_writer = MagicMock()
            saver.safetensors_writer.save_prefix = "quant_model_weights"
            
            saver.post_run()
            
            # Verify merge_ranks was called (barrier is called inside merge_ranks)
            mock_barrier.assert_called()
            # Verify copy_files was mocked (not actually called to copy files)
            mock_copy_files.assert_called_once()
    
    @patch(
        'msmodelslim.app.quant_service.modelslim_v1.save.'
        'ascendv1_distributed.DistHelper'
    )
    def test_prepare_for_distributed(
        self, mock_dist_helper_class, temp_dir, mock_model, mock_adapter
    ):
        """Test prepare_for_distributed method."""
        dist_path = (
            'msmodelslim.app.quant_service.modelslim_v1.save.'
            'ascendv1_distributed.dist'
        )
        # 修改特定参数：未初始化状态
        with patch('torch.distributed.is_initialized', return_value=False), \
             patch(f'{dist_path}.is_initialized', return_value=False):
            mock_dist_helper = MagicMock()
            mock_dist_helper.get_shared_modules_slice.return_value = [
                "module1", "module2"
            ]
            mock_dist_helper_class.return_value = mock_dist_helper
            
            saver = TestDistributedAscendV1Saver.create_saver(
                temp_dir, mock_model, mock_adapter
            )
            
            request = BatchProcessRequest(name="test", module=mock_model)
            saver.prepare_for_distributed(request)
            
            assert saver.dist_helper is mock_dist_helper
            assert saver.shared_modules_slice == ["module1", "module2"]
    
    @patch(
        'msmodelslim.app.quant_service.modelslim_v1.save.'
        'ascendv1_distributed.DistHelper'
    )
    def test_preprocess_and_postprocess(
        self, mock_dist_helper_class, temp_dir, mock_model, mock_adapter
    ):
        """Test preprocess and postprocess methods."""
        mock_dist_helper = MagicMock()
        mock_dist_helper.get_shared_modules_slice.return_value = []
        mock_dist_helper_class.return_value = mock_dist_helper
        
        saver = TestDistributedAscendV1Saver.create_saver(
            temp_dir, mock_model, mock_adapter
        )
        
        request = BatchProcessRequest(name="test", module=mock_model)
        saver.preprocess(request)
        assert saver.dist_helper is mock_dist_helper
        
        # Test postprocess cleanup
        saver.postprocess(request)
        assert saver.dist_helper is None
