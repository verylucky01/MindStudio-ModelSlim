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

import sys
import argparse
import tempfile
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch, call

import torch
import pytest

from msmodelslim.model.hunyuan_video.model_adapter import (
    HunyuanVideoModelAdapter, InvalidModelError,
    SchemaValidateError, UnsupportedError
)


@pytest.fixture(autouse=True)
def mock_hyvideo_modules(monkeypatch):
    """统一模拟所有hunyuan video相关模块"""
    mock_modules = [
        'hyvideo', 'hyvideo.constants', 'hyvideo.modules.models',
        'hyvideo.inference', 'hyvideo.utils.file_utils'
    ]

    original_modules = {mod: sys.modules.get(mod) for mod in mock_modules}
    for module_path in mock_modules:
        sys.modules[module_path] = MagicMock()

    # 配置关键模块
    hyvideo_constants = sys.modules['hyvideo.constants']
    hyvideo_constants.PRECISIONS = ['fp32', 'fp16', 'bf16']
    hyvideo_constants.VAE_PATH = {'884-16c-hy': '/fake/vae/path'}
    hyvideo_constants.TEXT_ENCODER_PATH = {'llm': '/fake/text/encoder'}
    hyvideo_constants.TOKENIZER_PATH = {'llm': '/fake/tokenizer'}
    hyvideo_constants.PROMPT_TEMPLATE = ['dit-llm-encode', 'dit-llm-encode-video']

    hyvideo_models = sys.modules['hyvideo.modules.models']
    hyvideo_models.HUNYUAN_VIDEO_CONFIG = {
        'HYVideo-T/2-cfgdistill': Mock()
    }

    # 模拟HunyuanVideoSampler
    mock_sampler = MagicMock()
    mock_sampler.pipeline = MagicMock()
    mock_sampler.pipeline.transformer = MagicMock()
    mock_sampler.predict = MagicMock(return_value=MagicMock())

    hyvideo_inference = sys.modules['hyvideo.inference']
    hyvideo_inference.HunyuanVideoSampler = MagicMock()
    hyvideo_inference.HunyuanVideoSampler.from_pretrained = MagicMock(return_value=mock_sampler)

    yield

    # 恢复原始模块
    for mod, original in original_modules.items():
        if original is not None:
            sys.modules[mod] = original
        else:
            if mod in sys.modules:
                del sys.modules[mod]


@pytest.fixture
def mock_env(monkeypatch):
    """环境变量模拟"""
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("WORLD_SIZE", "1")
    monkeypatch.setenv("LOCAL_RANK", "0")


@pytest.fixture
def temp_model_dir():
    """创建临时模型目录"""
    with tempfile.TemporaryDirectory() as temp_dir:
        model_base = Path(temp_dir)
        # 创建必要的子目录结构
        (model_base / "hunyuan-video-t2v-720p" / "transformers").mkdir(parents=True)
        (model_base / "vae").mkdir(parents=True)
        (model_base / "text_encoder").mkdir(parents=True)
        (model_base / "clip-vit-large-patch14").mkdir(parents=True)

        # 创建必要的权重文件
        weight_file = model_base / "hunyuan-video-t2v-720p" / "transformers" / "mp_rank_00_model_states.pt"
        weight_file.touch()

        yield model_base


# ------------------------------ 适配器基础功能测试 ------------------------------
class TestHunyuanVideoModelAdapter:
    @staticmethod
    def test_initialization(adapter, temp_model_dir):
        """测试初始化"""
        assert adapter.model_type == "hunyuan_video"
        assert adapter.model_path == temp_model_dir

    @staticmethod
    def test_get_model_info(adapter):
        """测试模型信息获取"""
        assert adapter.get_model_type() == "hunyuan_video"
        assert adapter.get_model_pedigree() == "hunyuan_video"

    @staticmethod
    def test_handle_dataset(adapter):
        """测试数据集处理"""
        mock_dataset = [Mock(), Mock()]
        result = adapter.handle_dataset(mock_dataset)
        assert list(result) == mock_dataset

    @staticmethod
    def test_enable_kv_cache(adapter):
        """测试KV缓存启用"""
        mock_model = Mock()
        adapter.enable_kv_cache(mock_model, True)
        # 方法应无异常执行
        assert True

    @staticmethod
    def test_init_model_returns_transformer(adapter):
        """测试模型初始化返回transformer"""
        mock_transformer = Mock()
        adapter.transformer = mock_transformer
        result = adapter.init_model()
        assert result == {'': mock_transformer}

    @pytest.fixture
    def adapter(self, temp_model_dir):
        """创建适配器实例"""
        with patch("msmodelslim.model.hunyuan_video.model_adapter.HunyuanVideoModelAdapter._check_import_dependency"):
            adapter_instance = HunyuanVideoModelAdapter("hunyuan_video", temp_model_dir)

        # 设置基本的model_args
        adapter_instance.model_args = MagicMock()
        adapter_instance.model_args.model_base = str(temp_model_dir)
        adapter_instance.model_args.prompt = "test prompt"
        adapter_instance.model_args.video_size = (720, 1280)
        adapter_instance.model_args.video_length = 129
        adapter_instance.model_args.infer_steps = 50
        adapter_instance.model_args.batch_size = 1
        adapter_instance.model_args.seed = 42

        return adapter_instance

# ------------------------------ 模型前向传播测试 ------------------------------
class TestModelForward:
    @staticmethod
    @pytest.mark.parametrize(
        "mock_inputs",  # 参数名
        [
            # 测试用例1：字典输入
            {"input_tensor": torch.randn(1, 10, 512)},
            # 测试用例2：元组输入
            (torch.randn(1, 10, 512), torch.randn(1, 10)),
        ],
        # 可选：为每个用例命名，方便识别测试结果
        ids=["dict_input", "tuple_input"]
    )
    def test_generate_model_forward_with_different_inputs(adapter_with_model, mock_inputs):
        """测试不同类型输入（字典/元组）的前向传播"""
        mock_model = adapter_with_model.transformer

        def mock_to_device_func(x, _):
            return x

        with patch('msmodelslim.model.hunyuan_video.model_adapter.to_device') as mock_to_device, \
                patch('msmodelslim.model.hunyuan_video.model_adapter.TransformersForwardBreak', Exception), \
                patch('msmodelslim.model.hunyuan_video.model_adapter.dist') as mock_dist:
            mock_to_device.side_effect = mock_to_device_func
            mock_dist.is_initialized.return_value = False

            generator = adapter_with_model.generate_model_forward(mock_model, mock_inputs)
            assert generator is not None

    @staticmethod
    def test_generate_model_visit(adapter_with_model):
        """测试模型遍历生成器"""
        mock_model = adapter_with_model.transformer

        # 使用patch来模拟generated_decoder_layer_visit_func_with_keyword
        with (patch(
                'msmodelslim.model.hunyuan_video.model_adapter.generated_decoder_layer_visit_func_with_keyword')
                as mock_visit_func):
            mock_generator = MagicMock()
            mock_visit_func.return_value = mock_generator

            generator = adapter_with_model.generate_model_visit(mock_model)

            # 验证返回的是正确的生成器
            assert generator == mock_generator
            # 验证函数被正确调用
            mock_visit_func.assert_called_once_with(mock_model, keyword="streamblock")

    @pytest.fixture
    def adapter_with_model(self, temp_model_dir):
        """创建带有正确模拟transformer模型的适配器"""
        with patch("msmodelslim.model.hunyuan_video.model_adapter.HunyuanVideoModelAdapter._check_import_dependency"):
            adapter = HunyuanVideoModelAdapter("hunyuan_video", temp_model_dir)

        # 创建真实的transformer模拟结构
        mock_transformer = MagicMock()

        # 创建包含"streamblock"的模拟模块
        mock_streamblock_1 = MagicMock()
        mock_streamblock_2 = MagicMock()

        type(mock_streamblock_1).__name__ = "StreamBlock"
        type(mock_streamblock_2).__name__ = "StreamBlock"

        # 设置模块的named_modules返回值
        mock_modules = [
            ('blocks.0', mock_streamblock_1),
            ('blocks.1', mock_streamblock_2),
            ('embedding', MagicMock()),
            ('norm', MagicMock()),
        ]

        mock_transformer.named_modules.return_value = mock_modules

        # 设置forward方法，确保会调用注册的hook
        def transformer_forward(*args, **kwargs):
            # 模拟调用第一个streamblock的前向传播
            # 这会触发注册的forward_pre_hook
            return mock_streamblock_1.forward(*args, **kwargs)

        mock_transformer.forward = transformer_forward
        mock_transformer.__call__ = transformer_forward

        # 设置适配器的transformer
        adapter.transformer = mock_transformer

        return adapter

# ------------------------------ Pipeline加载测试 ------------------------------
class TestLoadPipeline:
    @staticmethod
    def test_normal_execution(mock_self, mock_env):
        """测试正常执行流程"""
        # 模拟HunyuanVideoSampler.from_pretrained的返回值
        mock_sampler = MagicMock()
        mock_sampler.pipeline = MagicMock()
        mock_sampler.pipeline.transformer = MagicMock()

        from hyvideo.inference import HunyuanVideoSampler
        HunyuanVideoSampler.from_pretrained.return_value = mock_sampler

        HunyuanVideoModelAdapter._load_pipeline(mock_self)

        # 验证HunyuanVideoSampler被正确调用
        HunyuanVideoSampler.from_pretrained.assert_called_once()

        # 验证transformer被设置
        assert mock_self.transformer is not None
        assert mock_self.hunyuan_video_sampler is not None

    @staticmethod
    def test_unsupported_ulysses_degree(mock_self, mock_env):
        """测试不支持的ulysses_degree"""
        mock_self.model_args.ulysses_degree = 2
        with pytest.raises(UnsupportedError):
            HunyuanVideoModelAdapter._load_pipeline(mock_self)

    @staticmethod
    def test_unsupported_ring_degree(mock_self, mock_env):
        """测试不支持的ring_degree"""
        mock_self.model_args.ring_degree = 2
        with pytest.raises(UnsupportedError):
            HunyuanVideoModelAdapter._load_pipeline(mock_self)

    @staticmethod
    def test_unsupported_vae_parallel(mock_self, mock_env):
        """测试不支持的vae_parallel"""
        mock_self.model_args.vae_parallel = True
        with pytest.raises(UnsupportedError):
            HunyuanVideoModelAdapter._load_pipeline(mock_self)

    @staticmethod
    def test_load_pipeline_integration(mock_self):
        """测试load_pipeline完整流程"""
        mock_self._load_pipeline = Mock()
        mock_self._setup_cache = Mock()

        HunyuanVideoModelAdapter.load_pipeline(mock_self)

        mock_self._load_pipeline.assert_called_once()
        mock_self._setup_cache.assert_called_once()

    @pytest.fixture
    def mock_self(self, temp_model_dir):
        """创建模拟的self对象"""
        mock = Mock()
        mock.model_args = Mock()
        mock.model_args.model_base = str(temp_model_dir)
        mock.model_args.ulysses_degree = 1
        mock.model_args.ring_degree = 1
        mock.model_args.vae_parallel = False
        mock._check_import_dependency = Mock()
        return mock

# ------------------------------ 参数验证测试 ------------------------------
class TestValidateArgs:
    @staticmethod
    def test_invalid_prompt_type(mock_self, base_args):
        """测试无效的prompt类型"""
        base_args.prompt = 123
        with patch('pathlib.Path.exists', return_value=True):
            with pytest.raises(SchemaValidateError):
                HunyuanVideoModelAdapter._validate_args(mock_self, base_args)

    @staticmethod
    def test_empty_prompt(mock_self, base_args):
        """测试空prompt"""
        base_args.prompt = ""
        with patch('pathlib.Path.exists', return_value=True):
            with pytest.raises(SchemaValidateError):
                HunyuanVideoModelAdapter._validate_args(mock_self, base_args)

    @staticmethod
    def test_invalid_model_base(mock_self, base_args):
        """测试无效的模型路径"""
        base_args.model_base = "/nonexistent/path"
        with patch('pathlib.Path.exists', return_value=False):
            with pytest.raises(SchemaValidateError):
                HunyuanVideoModelAdapter._validate_args(mock_self, base_args)

    @staticmethod
    def test_infer_steps_default(mock_self, base_args):
        """测试infer_steps默认值"""
        base_args.infer_steps = None
        with patch('pathlib.Path.exists', return_value=True):
            HunyuanVideoModelAdapter._validate_args(mock_self, base_args)
        assert base_args.infer_steps == 50

    @staticmethod
    def test_infer_steps_invalid_type(mock_self, base_args):
        """测试无效的infer_steps类型"""
        base_args.infer_steps = "50"
        with patch('pathlib.Path.exists', return_value=True):
            with pytest.raises(SchemaValidateError):
                HunyuanVideoModelAdapter._validate_args(mock_self, base_args)

    @staticmethod
    def test_infer_steps_negative(mock_self, base_args):
        """测试负的infer_steps"""
        base_args.infer_steps = -1
        with patch('pathlib.Path.exists', return_value=True):
            with pytest.raises(SchemaValidateError):
                HunyuanVideoModelAdapter._validate_args(mock_self, base_args)

    @staticmethod
    def test_batch_size_default(mock_self, base_args):
        """测试batch_size默认值"""
        base_args.batch_size = None
        with patch('pathlib.Path.exists', return_value=True):
            HunyuanVideoModelAdapter._validate_args(mock_self, base_args)
        assert base_args.batch_size == 1

    @staticmethod
    def test_batch_size_invalid_type(mock_self, base_args):
        """测试无效的batch_size类型"""
        base_args.batch_size = "1"
        with patch('pathlib.Path.exists', return_value=True):
            with pytest.raises(SchemaValidateError):
                HunyuanVideoModelAdapter._validate_args(mock_self, base_args)

    @staticmethod
    def test_batch_size_negative(mock_self, base_args):
        """测试负的batch_size"""
        base_args.batch_size = 0
        with patch('pathlib.Path.exists', return_value=True):
            with pytest.raises(SchemaValidateError):
                HunyuanVideoModelAdapter._validate_args(mock_self, base_args)

    @staticmethod
    def test_seed_default(mock_self, base_args):
        """测试seed默认值"""
        base_args.seed = None
        with patch('pathlib.Path.exists', return_value=True):
            HunyuanVideoModelAdapter._validate_args(mock_self, base_args)
        assert base_args.seed == 0  # 注意：代码中先设为0，然后如果是负数才随机

    @staticmethod
    def test_seed_negative_random(mock_self, base_args):
        """测试负seed时的随机生成"""
        with patch("msmodelslim.model.hunyuan_video.model_adapter.random.randint", return_value=999):
            base_args.seed = -5
            with patch('pathlib.Path.exists', return_value=True):
                HunyuanVideoModelAdapter._validate_args(mock_self, base_args)
            assert base_args.seed == 999

    @staticmethod
    def test_task_config_set_correctly(mock_self, base_args):
        """测试task_config参数被正确设置为'hunyuanvideo'"""
        with patch('pathlib.Path.exists', return_value=True):
            HunyuanVideoModelAdapter._validate_args(mock_self, base_args)

        # 验证task_config被正确设置
        assert hasattr(base_args, 'task_config')
        assert base_args.task_config == 'hunyuanvideo'

    @pytest.fixture
    def mock_self(self):
        """创建模拟的self对象"""
        mock = Mock()
        mock._check_import_dependency = Mock()
        return mock

    @pytest.fixture
    def base_args(self, temp_model_dir):
        """基础参数fixture"""

        class Args:
            def __init__(self):
                self.model_base = str(temp_model_dir)
                self.save_path = "/tmp/results"
                self.save_path_suffix = ""
                self.infer_steps = 50
                self.batch_size = 1
                self.seed = 42
                self.prompt = "test prompt"

            def __contains__(self, item):
                return hasattr(self, item)

        return Args()


class TestSetModelArgs:
    @staticmethod
    def test_valid_update(mock_self, base_override):
        """测试有效参数更新"""
        valid_config = base_override({
            "infer_steps": 60,
            "batch_size": 2
        })
        mock_self.set_model_args(valid_config)

        mock_self._validate_args.assert_called_once()

    @staticmethod
    def test_illegal_attr_raise_error(mock_self, base_override):
        """测试非法属性错误"""
        invalid_config = base_override({
            "infer_steps": 60,
            "illegal_attr": "invalid"
        })
        with pytest.raises(SchemaValidateError) as exc:
            mock_self.set_model_args(invalid_config)

        assert "illegal config attributes:" in str(exc.value)

    @staticmethod
    def test_skip_none_value(mock_self, base_override):
        """测试跳过None值"""
        none_config = base_override({
            "infer_steps": 60,
            "batch_size": None
        })
        mock_self.set_model_args(none_config)

        # 验证parser的parse_args被调用
        mock_self._get_parser.return_value.parse_args.assert_called_once()

    @staticmethod
    def test_video_size_skipped(mock_self, base_override):
        """测试video_size被跳过"""
        video_config = base_override({
            "video_size": (720, 1280),
            "infer_steps": 60
        })
        mock_self.set_model_args(video_config)

        # 验证parser的parse_args被调用
        mock_self._get_parser.return_value.parse_args.assert_called_once()
    @pytest.fixture
    def mock_self(self, temp_model_dir):
        """创建模拟的self对象"""
        with patch("msmodelslim.model.hunyuan_video.model_adapter.HunyuanVideoModelAdapter._check_import_dependency"):
            mock = HunyuanVideoModelAdapter('hunyuan_video', temp_model_dir)

        # 创建基本的model_args
        class ModelArgs:
            def __init__(self):
                self.model_base = str(temp_model_dir)
                self.prompt = "test prompt"
                self.video_size = (720, 1280)
                self.infer_steps = 50
                self.latent_channels = 16
                self.batch_size = 1  # 添加batch_size属性
                self.vae = "884-16c-hy"

        mock.model_args = ModelArgs()
        mock._validate_args = Mock()

        # 创建模拟的parser
        mock_parser = Mock()
        mock_parser.parse_args = Mock(return_value=mock.model_args)
        mock._get_parser = Mock(return_value=mock_parser)
        mock.sanity_check_args = Mock(return_value=mock.model_args)
        return mock

    @pytest.fixture
    def base_override(self):
        """基础配置覆盖类"""

        class Override:
            def __init__(self, config):
                self.config = config

            def __getitem__(self, key):
                return self.config[key]

            def keys(self):
                return self.config.keys()

        return Override

# ------------------------------ 校准推理测试 ------------------------------
class TestRunCalibInference:
    @staticmethod
    def test_run_calib_inference_success(mock_self):
        """测试校准推理成功执行"""
        HunyuanVideoModelAdapter.run_calib_inference(mock_self)

        # 验证predict被调用
        mock_self.hunyuan_video_sampler.predict.assert_called_once()

        # 验证日志输出
        from msmodelslim.model.hunyuan_video.model_adapter import logging
        logging.info.assert_called_once()

    @pytest.fixture
    def mock_self(self):
        """创建模拟的self对象"""
        mock = Mock()

        # 模拟model_args
        mock.model_args = Mock(
            prompt="test prompt",
            video_size=(720, 1280),
            video_length=129,
            seed=42,
            neg_prompt=None,
            infer_steps=50,
            cfg_scale=1.0,
            num_videos=1,
            flow_shift=7.0,
            batch_size=1,
            embedded_cfg_scale=6.0
        )

        # 模拟hunyuan_video_sampler
        mock.hunyuan_video_sampler = Mock()
        mock.hunyuan_video_sampler.predict = Mock(return_value=Mock())

        return mock

    @pytest.fixture(autouse=True)
    def mock_dependencies(self):
        """模拟依赖"""
        with patch('msmodelslim.model.hunyuan_video.model_adapter.torch') as mock_torch, \
                patch('msmodelslim.model.hunyuan_video.model_adapter.tqdm') as mock_tqdm, \
                patch('msmodelslim.model.hunyuan_video.model_adapter.time') as mock_time, \
                patch('msmodelslim.model.hunyuan_video.model_adapter.logging'):
            mock_tqdm.return_value.__iter__.return_value = [1]
            mock_time.time.side_effect = [1.0, 3.0]  # begin=1.0, end=3.0
            mock_stream = Mock()
            mock_torch.npu.Stream.return_value = mock_stream

            yield

# ------------------------------ 量化应用测试 ------------------------------
class TestApplyQuantization:
    @staticmethod
    def test_apply_quantization_success(mock_self, process_func):
        """测试量化应用成功"""

        # 模拟no_sync方法
        class MockNoSync:
            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

        mock_self.no_sync = Mock(return_value=MockNoSync())

        HunyuanVideoModelAdapter.apply_quantization(mock_self, process_func)

        # 验证设备移动
        mock_self.transformer.named_modules.assert_called_once()
        process_func.assert_called_once()

    @pytest.fixture
    def mock_self(self):
        """创建包含transformer结构的模拟self对象"""
        mock = Mock()

        # 创建模拟的transformer模块
        module_embedding = Mock()
        module_block = Mock()
        module_norm = Mock()

        transformer = Mock()
        transformer.named_modules.return_value = [
            ('embedding', module_embedding),
            ('blocks.0', module_block),
            ('norm', module_norm)
        ]
        mock.transformer = transformer

        return mock

    @pytest.fixture
    def process_func(self):
        """模拟的处理函数"""
        return Mock()

# ------------------------------ Cache设置测试 ------------------------------
class TestSetupCache:
    @staticmethod
    def test_setup_cache_basic(mock_self):
        """测试基础cache设置"""
        with patch('mindiesd.CacheConfig') as mock_cache_config, \
                patch('mindiesd.CacheAgent') as mock_cache_agent:
            # 确保CacheConfig可以被实例化
            mock_cache_config.return_value = MagicMock()
            mock_cache_agent.return_value = MagicMock()

            HunyuanVideoModelAdapter._setup_cache(mock_self)

            # 验证CacheConfig被调用
            assert mock_cache_config.call_count >= 2

    @staticmethod
    def test_setup_cache_with_attention_cache(mock_self):
        """测试启用attention cache"""
        mock_self.model_args.use_attentioncache = True
        mock_self.model_args.start_step = 9
        mock_self.model_args.attentioncache_interval = 3
        mock_self.model_args.end_step = 47

        with patch('mindiesd.CacheConfig') as mock_cache_config, \
                patch('mindiesd.CacheAgent') as mock_cache_agent:
            # 确保CacheConfig可以被实例化
            mock_cache_config.return_value = MagicMock()
            mock_cache_agent.return_value = MagicMock()

            HunyuanVideoModelAdapter._setup_cache(mock_self)

            # 验证CacheConfig被调用时包含attention cache参数
            call_args_list = mock_cache_config.call_args_list
            attention_cache_calls = [
                call
                for call in call_args_list
                if call[1].get('method') == 'attention_cache'
            ]
            assert len(attention_cache_calls) >= 2

    @pytest.fixture
    def mock_self(self):
        """创建模拟的self对象"""
        mock = Mock()
        mock.model_args = Mock()
        mock.model_args.use_cache = False
        mock.model_args.use_cache_double = False
        mock.model_args.use_attentioncache = False
        mock.model_args.infer_steps = 50

        # 模拟transformer结构
        mock.transformer = Mock()
        mock.transformer.single_blocks = [Mock() for _ in range(10)]
        mock.transformer.double_blocks = [Mock() for _ in range(10)]

        # 模拟mindiesd模块
        mock_mindiesd = Mock()
        mock_mindiesd.CacheConfig = Mock()
        mock_mindiesd.CacheAgent = Mock()

        # 使用patch.dict模拟sys.modules
        with patch.dict('sys.modules',
                        {'mindiesd': mock_mindiesd}):
            yield mock

# ------------------------------ 参数解析器测试 ------------------------------
class TestGetParser:
    @staticmethod
    def test_get_parser_structure(mock_self):
        """测试参数解析器结构"""
        # 创建一个真实的parser用于测试
        parser = argparse.ArgumentParser(description="HunyuanVideo inference script")

        # 模拟每个add方法，让它们返回这个parser
        mock_self._HunyuanVideoModelAdapter__add_device_args = Mock(return_value=parser)
        mock_self._HunyuanVideoModelAdapter__add_network_args = Mock(return_value=parser)
        mock_self._HunyuanVideoModelAdapter__add_extra_models_args = Mock(return_value=parser)
        mock_self._HunyuanVideoModelAdapter__add_denoise_schedule_args = Mock(return_value=parser)
        mock_self._HunyuanVideoModelAdapter__add_inference_args = Mock(return_value=parser)
        mock_self._HunyuanVideoModelAdapter__add_parallel_args = Mock(return_value=parser)
        mock_self._HunyuanVideoModelAdapter__add_ditcache_args = Mock(return_value=parser)
        mock_self._HunyuanVideoModelAdapter__add_attentioncache_args = Mock(return_value=parser)
        mock_self._HunyuanVideoModelAdapter__add_quant_args = Mock(return_value=parser)

        result = HunyuanVideoModelAdapter._get_parser(mock_self)

        assert isinstance(result, argparse.ArgumentParser)

        # 验证所有add方法都被调用
        mock_self._HunyuanVideoModelAdapter__add_device_args.assert_called_once()
        mock_self._HunyuanVideoModelAdapter__add_network_args.assert_called_once()
        mock_self._HunyuanVideoModelAdapter__add_extra_models_args.assert_called_once()
        mock_self._HunyuanVideoModelAdapter__add_denoise_schedule_args.assert_called_once()
        mock_self._HunyuanVideoModelAdapter__add_inference_args.assert_called_once()
        mock_self._HunyuanVideoModelAdapter__add_parallel_args.assert_called_once()
        mock_self._HunyuanVideoModelAdapter__add_ditcache_args.assert_called_once()
        mock_self._HunyuanVideoModelAdapter__add_attentioncache_args.assert_called_once()
        mock_self._HunyuanVideoModelAdapter__add_quant_args.assert_called_once()

    @staticmethod
    def test_parser_add_methods_called(mock_self):
        """测试所有add方法被调用"""
        # 创建模拟方法
        mock_device = Mock()
        mock_network = Mock()
        mock_extra = Mock()
        mock_denoise = Mock()
        mock_inference = Mock()
        mock_parallel = Mock()
        mock_ditcache = Mock()
        mock_attentioncache = Mock()
        mock_quant = Mock()

        # 绑定模拟方法到mock_self
        mock_self._HunyuanVideoModelAdapter__add_device_args = mock_device
        mock_self._HunyuanVideoModelAdapter__add_network_args = mock_network
        mock_self._HunyuanVideoModelAdapter__add_extra_models_args = mock_extra
        mock_self._HunyuanVideoModelAdapter__add_denoise_schedule_args = mock_denoise
        mock_self._HunyuanVideoModelAdapter__add_inference_args = mock_inference
        mock_self._HunyuanVideoModelAdapter__add_parallel_args = mock_parallel
        mock_self._HunyuanVideoModelAdapter__add_ditcache_args = mock_ditcache
        mock_self._HunyuanVideoModelAdapter__add_attentioncache_args = mock_attentioncache
        mock_self._HunyuanVideoModelAdapter__add_quant_args = mock_quant

        # 调用方法
        HunyuanVideoModelAdapter._get_parser(mock_self)

        # 验证所有add方法都被调用
        mock_device.assert_called_once()
        mock_network.assert_called_once()
        mock_extra.assert_called_once()
        mock_denoise.assert_called_once()
        mock_inference.assert_called_once()
        mock_parallel.assert_called_once()
        mock_ditcache.assert_called_once()
        mock_attentioncache.assert_called_once()
        mock_quant.assert_called_once()

    @pytest.fixture
    def mock_self(self):
        """创建模拟的self对象"""
        mock = Mock()
        mock._check_import_dependency = Mock()
        return mock

# ------------------------------ 依赖检查测试 ------------------------------
class TestCheckImportDependency:
    @staticmethod
    def test_successful_import(mock_self):
        """测试依赖导入成功"""
        # 由于使用了fixture模拟，导入应该成功
        HunyuanVideoModelAdapter._check_import_dependency(mock_self)

    @staticmethod
    def test_import_failure(mock_self, monkeypatch):
        """测试依赖导入失败"""
        # 临时移除hyvideo模块
        original_hyvideo = sys.modules.get('hyvideo')
        if 'hyvideo' in sys.modules:
            monkeypatch.delitem(sys.modules, 'hyvideo')

        try:
            with pytest.raises(ImportError) as exc_info:
                HunyuanVideoModelAdapter._check_import_dependency(mock_self)
            assert "Failed to import required components from hunyuanvideo" in str(exc_info.value)
        finally:
            # 恢复模块
            if original_hyvideo is not None:
                sys.modules['hyvideo'] = original_hyvideo

    @pytest.fixture
    def mock_self(self):
        return Mock()


# ------------------------------ 参数检查测试 ------------------------------
class TestSanityCheckArgs:
    @staticmethod
    def test_valid_vae_pattern(mock_self):
        """测试有效的VAE模式"""
        args = argparse.Namespace()
        args.vae = "884-16c-hy"
        args.latent_channels = 16

        result = HunyuanVideoModelAdapter._HunyuanVideoModelAdapter__sanity_check_args(mock_self, args)
        assert result == args

    @staticmethod
    def test_invalid_vae_pattern(mock_self):
        """测试无效的VAE模式"""
        args = argparse.Namespace()
        args.vae = "invalid-pattern"
        args.latent_channels = 16

        with pytest.raises(SchemaValidateError):
            HunyuanVideoModelAdapter._HunyuanVideoModelAdapter__sanity_check_args(mock_self, args)

    @staticmethod
    def test_latent_channels_mismatch(mock_self):
        """测试潜在通道不匹配"""
        args = argparse.Namespace()
        args.vae = "884-16c-hy"  # 通道数为16
        args.latent_channels = 32  # 不匹配

        with pytest.raises(SchemaValidateError):
            HunyuanVideoModelAdapter._HunyuanVideoModelAdapter__sanity_check_args(mock_self, args)

    @staticmethod
    def test_auto_latent_channels(mock_self):
        """测试自动设置潜在通道"""
        args = argparse.Namespace()
        args.vae = "884-16c-hy"
        args.latent_channels = None

        result = HunyuanVideoModelAdapter._HunyuanVideoModelAdapter__sanity_check_args(mock_self, args)
        assert result.latent_channels == 16

    @pytest.fixture
    def mock_self(self):
        return Mock()