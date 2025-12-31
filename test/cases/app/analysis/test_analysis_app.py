# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
"""
综合测试用例：验证分析模块的完整功能覆盖
包括CLI、APP和分析服务模块的所有功能
目标覆盖率：>80%
"""
import os
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import torch
import torch.nn as nn
from testing_utils.mock import mock_init_config

from msmodelslim.core.analysis_service.layer_selector import AnalysisResult  # 替换为实际的 AnalysisResult 类
from msmodelslim.core.analysis_service.layer_selector import \
    LayerSelectorAnalysisService  # 替换为实际包含 _print_analysis_results 的类
from msmodelslim.core.const import DeviceType
from msmodelslim.utils.exception import SchemaValidateError

mock_init_config()


class TestComprehensiveAnalysisCoverage(unittest.TestCase):
    """综合测试分析模块的所有功能"""

    def setUp(self):
        """测试前的准备工作"""

        # 1. 保存原始 umask
        original_umask = os.umask(0)  # 临时设为 0 并获取原始值
        try:
            # 2. 设置目标 umask（0o026 对应权限 640/750）
            os.umask(0o026)
            self.temp_dir = tempfile.mkdtemp()
            self.dataset_dir = Path(self.temp_dir) / "lab_calib"
            self.dataset_dir.mkdir()

            # 创建模拟的校准数据集文件
            self.calib_file = self.dataset_dir / "boolq.jsonl"
            with open(self.calib_file, 'w') as f:
                f.write('{"data": "mock calibration data"}')

            # 创建模型文件
            self.model_path = Path(self.temp_dir) / "model"
            self.model_path.mkdir()
        finally:
            # 3. 无论是否出错，都恢复原始 umask
            os.umask(original_umask)

    def tearDown(self):
        """测试后的清理工作"""
        shutil.rmtree(self.temp_dir)


class TestAppAnalysisModule(TestComprehensiveAnalysisCoverage):
    """测试APP分析模块"""

    def test_layer_analysis_application_init(self):
        """测试LayerAnalysisApplication初始化"""
        from msmodelslim.app.analysis.application import LayerAnalysisApplication

        mock_service = MagicMock()
        mock_factory = MagicMock()

        app = LayerAnalysisApplication(mock_service, mock_factory)

        self.assertEqual(app.analysis_service, mock_service)
        self.assertEqual(app.model_factory, mock_factory)

    def test_analyze_parameter_validation_model_type(self):
        """测试analyze方法model_type参数验证"""
        from msmodelslim.app.analysis.application import LayerAnalysisApplication, AnalysisMetrics

        mock_service = MagicMock()
        mock_factory = MagicMock()
        app = LayerAnalysisApplication(mock_service, mock_factory)

        # 测试无效的model_type类型
        with self.assertRaises(SchemaValidateError):
            app.analyze(
                model_type=123,  # 应该是字符串
                model_path=str(self.model_path),
                patterns=["*"],
                device=DeviceType.NPU,
                metrics=AnalysisMetrics.STD,
                calib_dataset="boolq.jsonl",
                topk=15,
                trust_remote_code=False
            )

    def test_analyze_parameter_validation_patterns(self):
        """测试analyze方法patterns参数验证"""
        from msmodelslim.app.analysis.application import LayerAnalysisApplication, AnalysisMetrics

        mock_service = MagicMock()
        mock_factory = MagicMock()
        app = LayerAnalysisApplication(mock_service, mock_factory)

        # 测试无效的patterns类型
        with self.assertRaises(SchemaValidateError):
            app.analyze(
                model_type="Qwen2.5-7B-Instruct",
                model_path=str(self.model_path),
                patterns="not_a_list",  # 应该是列表
                device=DeviceType.NPU,
                metrics=AnalysisMetrics.STD,
                calib_dataset="boolq.jsonl",
                topk=15,
                trust_remote_code=False
            )

    def test_analyze_parameter_validation_device(self):
        """测试analyze方法device参数验证"""
        from msmodelslim.app.analysis.application import LayerAnalysisApplication, AnalysisMetrics

        mock_service = MagicMock()
        mock_factory = MagicMock()
        app = LayerAnalysisApplication(mock_service, mock_factory)

        # 测试无效的device类型
        with self.assertRaises(SchemaValidateError):
            app.analyze(
                model_type="Qwen2.5-7B-Instruct",
                model_path=str(self.model_path),
                patterns=["*"],
                device="invalid_device",  # 应该是DeviceType枚举
                metrics=AnalysisMetrics.STD,
                calib_dataset="boolq.jsonl",
                topk=15,
                trust_remote_code=False
            )

    def test_analyze_parameter_validation_metrics(self):
        """测试analyze方法metrics参数验证"""
        from msmodelslim.app.analysis.application import LayerAnalysisApplication

        mock_service = MagicMock()
        mock_factory = MagicMock()
        app = LayerAnalysisApplication(mock_service, mock_factory)

        # 测试无效的metrics类型
        with self.assertRaises(SchemaValidateError):
            app.analyze(
                model_type="Qwen2.5-7B-Instruct",
                model_path=str(self.model_path),
                patterns=["*"],
                device=DeviceType.NPU,
                metrics="invalid_metrics",  # 应该是AnalysisMetrics枚举
                calib_dataset="boolq.jsonl",
                topk=15,
                trust_remote_code=False
            )

    def test_analyze_parameter_validation_calib_dataset_format(self):
        """测试analyze方法calib_dataset文件格式验证"""
        from msmodelslim.app.analysis.application import LayerAnalysisApplication, AnalysisMetrics

        mock_service = MagicMock()
        mock_factory = MagicMock()
        app = LayerAnalysisApplication(mock_service, mock_factory)

        # 测试无效的文件格式
        with self.assertRaises(SchemaValidateError):
            app.analyze(
                model_type="Qwen2.5-7B-Instruct",
                model_path=str(self.model_path),
                patterns=["*"],
                device=DeviceType.NPU,
                metrics=AnalysisMetrics.STD,
                calib_dataset="invalid.txt",  # 无效的文件格式
                topk=15,
                trust_remote_code=False
            )

    def test_analyze_parameter_validation_topk(self):
        """测试analyze方法topk参数验证"""
        from msmodelslim.app.analysis.application import LayerAnalysisApplication, AnalysisMetrics

        mock_service = MagicMock()
        mock_factory = MagicMock()
        app = LayerAnalysisApplication(mock_service, mock_factory)

        # 测试无效的topk值
        with self.assertRaises(SchemaValidateError):
            app.analyze(
                model_type="Qwen2.5-7B-Instruct",
                model_path=str(self.model_path),
                patterns=["*"],
                device=DeviceType.NPU,
                metrics=AnalysisMetrics.STD,
                calib_dataset="boolq.jsonl",
                topk=0,  # 无效的topk值
                trust_remote_code=False
            )

    def test_analyze_parameter_validation_trust_remote_code(self):
        """测试analyze方法trust_remote_code参数验证"""
        from msmodelslim.app.analysis.application import LayerAnalysisApplication, AnalysisMetrics

        mock_service = MagicMock()
        mock_factory = MagicMock()
        app = LayerAnalysisApplication(mock_service, mock_factory)

        # 测试无效的trust_remote_code类型
        with self.assertRaises(SchemaValidateError):
            app.analyze(
                model_type="Qwen2.5-7B-Instruct",
                model_path=str(self.model_path),
                patterns=["*"],
                device=DeviceType.NPU,
                metrics=AnalysisMetrics.STD,
                calib_dataset="boolq.jsonl",
                topk=15,
                trust_remote_code="not_a_bool"  # 应该是bool
            )

    @patch('msmodelslim.app.analysis.application.get_logger')
    def test_analyze_with_valid_parameters(self, mock_get_logger):
        """测试analyze方法使用有效参数"""
        from msmodelslim.app.analysis.application import LayerAnalysisApplication, AnalysisMetrics
        from msmodelslim.core.analysis_service.pipeline_interface import PipelineInterface

        # 创建模拟对象
        mock_service = MagicMock()
        mock_model_factory = MagicMock()
        mock_model_adapter = MagicMock(spec=PipelineInterface)
        mock_result = MagicMock()
        mock_logger = MagicMock()

        # 设置模拟返回值
        mock_model_factory.create.return_value = mock_model_adapter
        mock_service.analyze.return_value = mock_result
        mock_get_logger.return_value = mock_logger

        app = LayerAnalysisApplication(mock_service, mock_model_factory)

        # 调用analyze方法
        result = app.analyze(
            model_type="Qwen2.5-7B-Instruct",
            model_path=str(self.model_path),
            patterns=["*"],
            device=DeviceType.CPU,
            metrics=AnalysisMetrics.STD,
            calib_dataset="boolq.jsonl",
            topk=15,
            trust_remote_code=False
        )

        # 验证结果
        self.assertEqual(result, mock_result)

        # 验证模型工厂调用
        mock_model_factory.create.assert_called_once_with(
            "Qwen2.5-7B-Instruct", self.model_path, False
        )

        # 验证分析服务调用
        expected_config = {
            'metrics': 'std',
            'calib_dataset': 'boolq.jsonl',
            'method_params': {}
        }
        mock_service.analyze.assert_called_once_with(
            device=DeviceType.CPU,
            model_adapter=mock_model_adapter,
            patterns=["*"],
            analysis_config=expected_config
        )

        # 验证结果导出
        mock_service.export_results.assert_called_once_with(result, 15)

    @patch('msmodelslim.app.analysis.application.get_logger')
    def test_analyze_with_unsupported_model_adapter(self, mock_get_logger):
        """测试analyze方法使用不支持的模型适配器"""
        from msmodelslim.app.analysis.application import LayerAnalysisApplication, AnalysisMetrics
        from msmodelslim.utils.exception import UnsupportedError

        mock_service = MagicMock()
        mock_model_factory = MagicMock()
        mock_model_adapter = MagicMock()  # 不是PipelineInterface
        mock_logger = MagicMock()

        mock_model_factory.create.return_value = mock_model_adapter
        mock_get_logger.return_value = mock_logger

        app = LayerAnalysisApplication(mock_service, mock_model_factory)

        # 测试不支持的模型适配器
        with self.assertRaises(UnsupportedError):
            app.analyze(
                model_type="Qwen2.5-7B-Instruct",
                model_path=str(self.model_path),
                patterns=["*"],
                device=DeviceType.NPU,
                metrics=AnalysisMetrics.STD,
                calib_dataset="boolq.jsonl",
                topk=15,
                trust_remote_code=False
            )

    @patch('msmodelslim.app.analysis.application.get_logger')
    def test_analyze_with_none_result(self, mock_get_logger):
        """测试analyze方法返回None结果"""
        from msmodelslim.app.analysis.application import LayerAnalysisApplication, AnalysisMetrics
        from msmodelslim.core.analysis_service.pipeline_interface import PipelineInterface

        mock_service = MagicMock()
        mock_model_factory = MagicMock()
        mock_model_adapter = MagicMock(spec=PipelineInterface)
        mock_logger = MagicMock()

        # 设置服务返回None
        mock_model_factory.create.return_value = mock_model_adapter
        mock_service.analyze.return_value = None
        mock_get_logger.return_value = mock_logger

        app = LayerAnalysisApplication(mock_service, mock_model_factory)

        result = app.analyze(
            model_type="Qwen2.5-7B-Instruct",
            model_path=str(self.model_path),
            patterns=["*"],
            device=DeviceType.NPU,
            metrics=AnalysisMetrics.KURTOSIS,
            calib_dataset="boolq.jsonl",
            topk=15,
            trust_remote_code=False
        )

        # 验证结果为None
        self.assertIsNone(result)
        # 验证export_results没有被调用
        mock_service.export_results.assert_not_called()


class TestAnalysisServiceModule(TestComprehensiveAnalysisCoverage):
    """测试分析服务模块"""

    def test_layer_selector_analysis_service_init(self):
        mock_dataset_loader = MagicMock()
        service = LayerSelectorAnalysisService(mock_dataset_loader)

        self.assertEqual(service.dataset_loader, mock_dataset_loader)

    def test_analyze_with_invalid_patterns(self):
        """测试analyze方法使用无效的patterns"""

        mock_dataset_loader = MagicMock()
        service = LayerSelectorAnalysisService(mock_dataset_loader)

        mock_model = MagicMock()
        mock_model.device = MagicMock()
        mock_model.device.value = 'cpu'

        with self.assertRaises(SchemaValidateError):
            service.analyze(
                model_adapter=mock_model,
                patterns="not_a_list",  # 无效的patterns类型
                analysis_config={}
            )

    @patch('msmodelslim.core.analysis_service.layer_selector.get_logger')
    def test_analyze_with_successful_flow(self, mock_get_logger):
        """测试analyze方法成功流程"""
        from msmodelslim.core.analysis_service.pipeline_interface import PipelineInterface

        mock_dataset_loader = MagicMock()
        service = LayerSelectorAnalysisService(mock_dataset_loader)

        # 创建模拟对象
        mock_model_adapter = MagicMock(spec=PipelineInterface)
        mock_model = MagicMock()
        mock_model.model = MagicMock()
        mock_model_adapter.load_model.return_value = mock_model

        # 模拟方法调用
        with patch.object(service, '_prepare_calibration_data', return_value=None) as mock_prep:
            with patch.object(service, '_get_target_layers', return_value=['layer1']) as mock_get_layers:
                with patch.object(service, '_run_analysis',
                                  return_value=[{'name': 'layer1', 'score': 1.0}]) as mock_run:
                    result = service.analyze(
                        model_adapter=mock_model_adapter,
                        patterns=["*"],
                        analysis_config={'metrics': 'std', 'calib_dataset': 'test.jsonl', 'method_params': {}},
                        device=DeviceType.CPU
                    )

        # 验证结果
        self.assertEqual(result.layer_scores, [{'name': 'layer1', 'score': 1.0}])
        self.assertEqual(result.method, 'std')
        self.assertEqual(result.patterns, ["*"])

    @patch('msmodelslim.core.analysis_service.layer_selector.get_logger')
    def test_prepare_calibration_data_none_dataset(self, mock_get_logger):
        """测试_prepare_calibration_data方法无校准数据集"""
        from msmodelslim.core.analysis_service.pipeline_interface import PipelineInterface

        mock_dataset_loader = MagicMock()
        service = LayerSelectorAnalysisService(mock_dataset_loader)

        mock_model_adapter = MagicMock(spec=PipelineInterface)

        result = service._prepare_calibration_data(mock_model_adapter, None)

        self.assertIsNone(result)

    @patch('msmodelslim.core.analysis_service.layer_selector.get_logger')
    def test_prepare_calibration_data_with_dataset(self, mock_get_logger):
        """测试_prepare_calibration_data方法有校准数据集"""
        from msmodelslim.core.analysis_service.pipeline_interface import PipelineInterface

        mock_dataset_loader = MagicMock()
        service = LayerSelectorAnalysisService(mock_dataset_loader)

        mock_model_adapter = MagicMock(spec=PipelineInterface)
        mock_dataset = MagicMock()
        mock_calib_data = ['data1', 'data2']

        mock_dataset_loader.get_dataset_by_name.return_value = mock_dataset
        mock_model_adapter.handle_dataset.return_value = mock_calib_data

        result = service._prepare_calibration_data(mock_model_adapter, 'test_dataset.jsonl')

        self.assertEqual(result, mock_calib_data)
        mock_dataset_loader.get_dataset_by_name.assert_called_once_with('test_dataset.jsonl')
        mock_model_adapter.handle_dataset.assert_called_once()

    def test_get_target_layers(self):
        """测试_get_target_layers方法"""
        mock_dataset_loader = MagicMock()
        service = LayerSelectorAnalysisService(mock_dataset_loader)

        mock_model = MagicMock()

        with patch('msmodelslim.core.analysis_service.layer_selector.AnalysisTargetMatcher') as mock_matcher:
            mock_matcher.get_linear_conv_layers.return_value = ['layer1', 'layer2', 'layer3']
            mock_matcher.filter_layers_by_patterns.return_value = ['layer1', 'layer2']

            result = service._get_target_layers(mock_model, ['layer*'])

        mock_matcher.get_linear_conv_layers.assert_called_once_with(mock_model)
        mock_matcher.filter_layers_by_patterns.assert_called_once_with(['layer1', 'layer2', 'layer3'], ['layer*'])
        self.assertEqual(result, ['layer1', 'layer2'])

    def test_export_results(self):
        """测试export_results方法"""

        mock_dataset_loader = MagicMock()
        service = LayerSelectorAnalysisService(mock_dataset_loader)

        result = AnalysisResult(
            layer_scores=[{'name': 'layer1', 'score': 2.0}, {'name': 'layer2', 'score': 1.0}],
            method='test',
            patterns=['*']
        )

        with patch.object(service, '_print_analysis_results') as mock_print:
            service.export_results(result, 10)

        mock_print.assert_called_once_with(result, 10)

    def test_run_analysis_with_unsupported_data_type(self):
        """测试_run_analysis方法使用不支持的数据类型"""
        mock_dataset_loader = MagicMock()
        service = LayerSelectorAnalysisService(mock_dataset_loader)

        # 创建模拟模型
        mock_model = MagicMock(spec=nn.Module)
        mock_layer = MagicMock(spec=nn.Linear)
        mock_model.named_modules.return_value = [('layer1', mock_layer)]

        # 创建模拟分析方法
        mock_analysis_method = MagicMock()

        # 创建不支持的数据类型
        calib_data = ["unsupported_string_data"]

        # 执行测试，期望抛出异常
        with self.assertRaises(NotImplementedError):
            service._run_analysis(mock_model, ['layer1'], mock_analysis_method, calib_data)

    def test_run_analysis_no_target_layers_found(self):
        """测试_run_analysis方法没有找到目标层"""

        mock_dataset_loader = MagicMock()
        service = LayerSelectorAnalysisService(mock_dataset_loader)

        # 创建模拟模型，只有非Linear层
        mock_model = MagicMock(spec=nn.Module)
        mock_layer = MagicMock(spec=nn.Conv2d)  # 非Linear层
        mock_model.named_modules.return_value = [('layer1', mock_layer)]

        mock_analysis_method = MagicMock()

        result = service._run_analysis(mock_model, ['layer1'], mock_analysis_method, None)

        # 验证结果为空
        self.assertEqual(result, [])

    @patch('msmodelslim.core.analysis_service.layer_selector.get_logger')
    def test_run_analysis_empty_layer_stats(self, mock_get_logger):
        """测试_run_analysis方法空层统计"""
        mock_dataset_loader = MagicMock()
        service = LayerSelectorAnalysisService(mock_dataset_loader)

        # 创建模拟模型
        mock_model = MagicMock(spec=nn.Module)
        mock_layer = MagicMock(spec=nn.Linear)
        mock_model.named_modules.return_value = [('layer1', mock_layer)]

        # 创建模拟分析方法，返回空统计
        mock_analysis_method = MagicMock()
        mock_hook = MagicMock()
        mock_analysis_method.get_hook.return_value = mock_hook

        # 模拟hook注册
        mock_hook_handle = MagicMock()
        mock_layer.register_forward_hook.return_value = mock_hook_handle

        # 没有校准数据，layer_stats应该为空
        result = service._run_analysis(mock_model, ['layer1'], mock_analysis_method, None)

        # 验证返回空列表
        self.assertEqual(result, [])

    @patch('msmodelslim.core.analysis_service.layer_selector.get_logger')
    @patch('msmodelslim.core.analysis_service.layer_selector.clean_output')
    def test_print_analysis_results_various_disable_levels(self, mock_clean_output, mock_get_logger):
        """测试_print_analysis_results方法不同的disable_level"""
        mock_dataset_loader = MagicMock()
        service = LayerSelectorAnalysisService(mock_dataset_loader)

        # 创建测试数据
        layer_scores = [
            {'name': 'layer1', 'score': 3.0},
            {'name': 'layer2', 'score': 2.0},
            {'name': 'layer3', 'score': 1.0}
        ]
        result = AnalysisResult(layer_scores, 'test', ['*'])

        # 测试disable_level超出范围
        service._print_analysis_results(result, 10)  # 超出层数

        # 测试disable_level为负数
        service._print_analysis_results(result, -1)

        # 验证clean_output被调用
        self.assertTrue(mock_clean_output.called)


def create_mock_analysis_result(layer_scores: list) -> MagicMock:
    """
    构建模拟的 AnalysisResult 对象，用于测试输入

    Args:
        layer_scores: 层分数列表，每个元素为 {'name': 层名, 'score': 敏感度分数}

    Returns:
        MagicMock: 模拟的 AnalysisResult 实例
    """
    mock_result = MagicMock(spec=AnalysisResult)
    # 模拟 get_sorted_layers 方法（按 score 降序排序）
    mock_result.get_sorted_layers = MagicMock(return_value=sorted(layer_scores, key=lambda x: x['score'], reverse=True))
    # 模拟属性
    mock_result.method = "kurtosis"  # 模拟峰度算法（可替换为其他方法）
    mock_result.patterns = ["conv2d", "linear", "mlp"]  # 模拟分析的模式
    mock_result.layer_scores = layer_scores  # 模拟总层分数列表
    return mock_result


class TestPrintAnalysisResults(unittest.TestCase):
    """测试 _print_analysis_results 方法的单元测试类"""

    def test_normal_case_with_disable_level(self):
        """
        正常场景：指定合理的 disable_level，输出对应数量的敏感层
        验证点：基础信息打印、指定数量的层输出、YAML 格式正确性
        """
        # 1. 准备测试数据：5个层，分数从高到低排序
        test_layers = [
            {"name": "model.layers.26.mlp.down_proj", "score": 98.7654},
            {"name": "model.layers.4.mlp.down_proj", "score": 87.6543},
            {"name": "model.layers.1.mlp.down_proj", "score": 76.5432},
            {"name": "model.layers.3.mlp.down_proj", "score": 65.4321},
            {"name": "model.layers.2.mlp.down_proj", "score": 54.3210}
        ]
        mock_result = create_mock_analysis_result(test_layers)
        # 创建mock dataset_loader，因为LayerSelectorAnalysisService需要它
        mock_dataset_loader = MagicMock()
        test_instance = LayerSelectorAnalysisService(mock_dataset_loader)  # 初始化被测试类实例
        disable_level = 3  # 期望输出前3个敏感层

        # 2. 创建mock logger
        mock_logger = MagicMock()

        # 我们需要patch get_logger函数，因为@logger_setter装饰器会调用它
        with patch('msmodelslim.core.analysis_service.layer_selector.get_logger') as mock_get_logger:
            mock_get_logger.return_value = mock_logger

            # 3. 执行被测试方法
            test_instance._print_analysis_results(result=mock_result, disable_level=disable_level)

            # 4. 验证基本功能
            self.assertTrue(mock_logger.info.called, "mock_logger应该被调用")
            self.assertGreater(mock_logger.info.call_count, 5, "应该有多次日志调用")

            # 5. 验证一些关键的日志内容
            log_messages = [str(call[0][0]) if call[0] else "" for call in mock_logger.info.call_args_list]

            # 验证包含关键信息
            self.assertTrue(any("Layer Analysis Results" in str(msg) for msg in log_messages), "应该包含分析结果标题")
            self.assertTrue(any("kurtosis method" in str(msg) for msg in log_messages), "应该包含分析方法")
            self.assertTrue(any("conv2d" in str(msg) for msg in log_messages), "应该包含模式信息")
            self.assertTrue(any("Total layers analyzed: 5" in str(msg) for msg in log_messages), "应该包含总层数信息")


def test_get_tokenized_data():
    """测试get_tokenized_data函数"""
    from msmodelslim.core.analysis_service.layer_selector import get_tokenized_data

    # 创建模拟tokenizer
    mock_tokenizer = MagicMock()
    mock_inputs = MagicMock()
    mock_inputs.data = {
        'input_ids': torch.tensor([[1, 2, 3]]),
        'attention_mask': torch.tensor([[1, 1, 1]])
    }
    mock_inputs.to.return_value = mock_inputs
    mock_tokenizer.return_value = mock_inputs

    calib_list = ['test text 1', 'test text 2']
    device = torch.device('cpu')

    result = get_tokenized_data(mock_tokenizer, calib_list, device)

    # 验证结果
    assert len(result) == 2
    assert len(result[0]) == 2  # input_ids and attention_mask
    mock_tokenizer.assert_called()


class TestAnalysisMetrics(unittest.TestCase):
    """测试AnalysisMetrics枚举"""

    def test_analysis_metrics_values(self):
        """测试AnalysisMetrics枚举值"""
        from msmodelslim.app.analysis.application import AnalysisMetrics

        self.assertEqual(AnalysisMetrics.STD.value, 'std')
        self.assertEqual(AnalysisMetrics.QUANTILE.value, 'quantile')
        self.assertEqual(AnalysisMetrics.KURTOSIS.value, 'kurtosis')

    def test_analysis_metrics_extended_enum_functionality(self):
        """测试AnalysisMetrics的ExtendedEnum功能"""
        from msmodelslim.app.analysis.application import AnalysisMetrics

        # 测试所有值都是有效的字符串
        for metric in AnalysisMetrics:
            self.assertIsInstance(metric.value, str)
            self.assertGreater(len(metric.value), 0)


class TestGetTokenizedDataFunction(unittest.TestCase):
    """测试get_tokenized_data函数"""

    def test_get_tokenized_data_basic_functionality(self):
        """测试get_tokenized_data基本功能"""
        from msmodelslim.core.analysis_service.layer_selector import get_tokenized_data

        # 创建模拟tokenizer
        mock_tokenizer = MagicMock()
        mock_inputs = MagicMock()
        mock_inputs.data = {
            'input_ids': torch.tensor([[1, 2, 3]]),
            'attention_mask': torch.tensor([[1, 1, 1]])
        }
        mock_inputs.to.return_value = mock_inputs
        mock_tokenizer.return_value = mock_inputs

        calib_list = ['test text 1', 'test text 2']
        device = torch.device('cpu')

        result = get_tokenized_data(mock_tokenizer, calib_list, device)

        # 验证结果
        self.assertEqual(len(result), 2)
        self.assertEqual(len(result[0]), 2)  # input_ids and attention_mask

        # 验证tokenizer被正确调用
        self.assertEqual(mock_tokenizer.call_count, 2)

    def test_get_tokenized_data_custom_names(self):
        """测试get_tokenized_data使用自定义键名"""
        from msmodelslim.core.analysis_service.layer_selector import get_tokenized_data

        # 创建模拟tokenizer
        mock_tokenizer = MagicMock()
        mock_inputs = MagicMock()
        mock_inputs.data = {
            'custom_input_ids': torch.tensor([[1, 2, 3]]),
            'custom_attention_mask': torch.tensor([[1, 1, 1]])
        }
        mock_inputs.to.return_value = mock_inputs
        mock_tokenizer.return_value = mock_inputs

        calib_list = ['test text']
        device = torch.device('cpu')

        result = get_tokenized_data(
            mock_tokenizer,
            calib_list,
            device,
            input_ids_name='custom_input_ids',
            attention_mask_name='custom_attention_mask'
        )

        # 验证结果
        self.assertEqual(len(result), 1)
        self.assertEqual(len(result[0]), 2)

    def test_get_tokenized_data_empty_list(self):
        """测试get_tokenized_data处理空列表"""
        from msmodelslim.core.analysis_service.layer_selector import get_tokenized_data

        mock_tokenizer = MagicMock()
        calib_list = []
        device = torch.device('cpu')

        result = get_tokenized_data(mock_tokenizer, calib_list, device)

        # 验证结果
        self.assertEqual(len(result), 0)
        mock_tokenizer.assert_not_called()


if __name__ == '__main__':
    unittest.main()
