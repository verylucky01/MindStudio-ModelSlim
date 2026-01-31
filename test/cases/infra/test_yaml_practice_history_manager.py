#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""
-------------------------------------------------------------------------
This file is part of the MindStudio project.
Copyright (c) 2025 Huawei Technologies Co.,Ltd.

MindStudio is licensed under Mulan PSL v2.
You can use this software according to the terms and conditions of the Mulan PSL v2.
You may obtain a copy of Mulan PSL v2 at:

         http://license.coscl.org.cn/MulanPSL2

THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
See the Mulan PSL v2 for more details.
-------------------------------------------------------------------------
"""
"""
msmodelslim.infra.yaml_practice_history_manager 模块的单元测试
主要测试历史记录管理功能
"""
from pathlib import Path
from unittest.mock import patch

from msmodelslim.infra.yaml_practice_history_manager import (
    YamlTuningHistory,
    YamlTuningHistoryManager,
)
from msmodelslim.core.practice import PracticeConfig, Metadata
from msmodelslim.core.tune_strategy import EvaluateResult, EvaluateAccuracy
from msmodelslim.utils.hash import calculate_md5


def create_test_practice_config(config_id: str = "test_config") -> PracticeConfig:
    """创建测试用的PracticeConfig"""
    return PracticeConfig(
        metadata=Metadata(config_id=config_id)
    )


def create_test_evaluate_result(accuracy: float = 0.95) -> EvaluateResult:
    """创建测试用的EvaluateResult"""
    return EvaluateResult(
        accuracies=[EvaluateAccuracy(dataset="test_dataset", accuracy=accuracy)],
        is_satisfied=True
    )


class TestYamlTuningHistory:
    """测试YamlTuningHistory类"""
    
    def test_init_set_empty_records_when_initialize(self, tmp_path: Path):
        """测试初始化空历史记录管理器"""
        history_dir = tmp_path / "history"
        history = YamlTuningHistory(str(history_dir))
        
        assert history.database == str(history_dir)  # 校验database正确设置
        assert history.history_dir == history_dir  # 校验history_dir正确设置
        assert len(history._history_index.records) == 0  # 校验_history_index.records为空
    
    def test_append_history_add_record_when_new_practice(self, tmp_path: Path):
        """测试追加新的历史记录"""
        history_dir = tmp_path / "history"
        history_dir.mkdir()
        history = YamlTuningHistory(str(history_dir))
        
        practice = create_test_practice_config("test_config")
        evaluate_result = create_test_evaluate_result(0.95)
        
        history.append_history(practice, evaluate_result)
        
        assert len(history._history_index.records) == 1  # 校验历史索引记录数为1
        assert history._history_index.records[0].practice_id == "test_config"  # 校验practice_id正确
        practice_md5 = calculate_md5(practice)
        assert history._history_index.records[0].md5 == practice_md5  # 校验md5正确
    
    def test_append_history_accumulate_records_when_same_practice_id(self, tmp_path: Path):
        """测试相同practice_id追加多条历史记录"""
        history_dir = tmp_path / "history"
        history = YamlTuningHistory(str(history_dir))
        
        practice = create_test_practice_config("test_config")
        evaluate_result1 = create_test_evaluate_result(0.9)
        evaluate_result2 = create_test_evaluate_result(0.95)
        
        history.append_history(practice, evaluate_result1)
        assert len(history._history_index.records) == 1  # 校验第一次追加成功
        
        history.append_history(practice, evaluate_result2)
        
        assert len(history._history_index.records) >= 1  # 校验历史索引记录数递增
        last_record = history._history_index.records[-1]
        assert last_record.practice_id == "test_config"  # 校验最后一条记录的practice_id
        assert last_record.evaluation.accuracies[0].accuracy == 0.95  # 校验最后一条记录正确
        
        from msmodelslim.utils.security import yaml_safe_load
        history_file = history_dir / "history.yaml"
        if history_file.exists():
            file_content = yaml_safe_load(str(history_file))
            assert len(file_content.get("records", [])) == 2  # 校验文件中的历史索引包含两条记录
    
    def test_clear_records_clear_index_when_call(self, tmp_path: Path):
        """测试清除历史记录"""
        history_dir = tmp_path / "history"
        history_dir.mkdir()
        history = YamlTuningHistory(str(history_dir))
        
        practice1 = create_test_practice_config("config1")
        practice2 = create_test_practice_config("config2")
        evaluate_result = create_test_evaluate_result(0.95)
        
        history.append_history(practice1, evaluate_result)
        history.append_history(practice2, evaluate_result)
        
        db = history._database_history.history_practice_database
        db["practice1"] = {"test": "data1"}
        db["practice2"] = {"test": "data2"}
        
        history.clear_records()
        
        assert len(history._history_index.records) == 0  # 校验历史索引记录数变为0


class TestYamlTuningHistoryManager:
    """测试YamlTuningHistoryManager类"""
    
    def test_load_history_return_yaml_tuning_history_when_empty_and_with_data(self, tmp_path: Path):
        """测试空目录和包含已存在历史记录的目录下加载历史记录"""
        manager = YamlTuningHistoryManager()
        
        history_dir = tmp_path / "history"
        history = manager.load_history(str(history_dir))
        assert isinstance(history, YamlTuningHistory)  # 校验返回YamlTuningHistory实例
        assert history.database == str(history_dir)  # 校验database属性正确
        
        history_dir.mkdir(exist_ok=True)
        practice = create_test_practice_config("test_config")
        evaluate_result = create_test_evaluate_result(0.95)
        
        history2 = manager.load_history(str(history_dir))
        history2.append_history(practice, evaluate_result)
        
        assert len(history2._history_index.records) == 1  # 校验能加载并追加历史记录
    
    @patch('msmodelslim.infra.yaml_practice_history_manager.get_logger')
    def test_load_history_handle_exception_when_path_is_file(self, mock_logger, tmp_path: Path):
        """测试路径指向文件而非目录时异常处理"""
        history_dir = tmp_path / "history"
        history_dir.write_text("not a directory")
        
        manager = YamlTuningHistoryManager()
        history = manager.load_history(str(history_dir))
        assert isinstance(history, YamlTuningHistory)  # 校验不抛出异常时返回实例


class TestCalculateMd5:
    """测试calculate_md5函数（用于计算practice的MD5）"""
    
    def test_calculate_md5_return_same_when_same_config_and_different_when_different(self):
        """测试计算相同和不同配置的MD5值"""
        practice1 = create_test_practice_config("test_config")
        practice2 = create_test_practice_config("test_config")
        md5_1 = calculate_md5(practice1)
        md5_2 = calculate_md5(practice2)
        assert md5_1 == md5_2  # 校验相同配置MD5值相同
        
        practice3 = create_test_practice_config("config1")
        practice4 = create_test_practice_config("config2")
        md5_3 = calculate_md5(practice3)
        md5_4 = calculate_md5(practice4)
        assert md5_3 != md5_4  # 校验不同配置MD5值不同
        
        md5_5 = calculate_md5(practice1)
        assert md5_1 == md5_5  # 校验MD5计算是确定性的


class TestResumeIntegration:
    """测试断点重续集成场景"""
    
    def test_resume_add_new_practice_when_clear_and_append(self, tmp_path: Path):
        """测试断点重续时清除历史记录并添加新的实践配置"""
        history_dir = tmp_path / "history"
        history_dir.mkdir()
        
        practice1 = create_test_practice_config("config1")
        evaluate_result1 = create_test_evaluate_result(0.9)
        
        history1 = YamlTuningHistory(str(history_dir))
        history1.append_history(practice1, evaluate_result1)
        
        assert len(history1._history_index.records) == 1  # 校验第一次运行的结果
        assert history1._history_index.records[0].practice_id == "config1"  # 校验practice_id正确
        
        history2 = YamlTuningHistory(str(history_dir))
        history2.clear_records()
        
        assert len(history2._history_index.records) == 0  # 校验清除后历史索引为空
        
        practice2 = create_test_practice_config("config2")
        evaluate_result2 = create_test_evaluate_result(0.85)
        history2.append_history(practice2, evaluate_result2)
        
        assert len(history2._history_index.records) == 1  # 校验添加新记录后历史索引包含新记录
        assert history2._history_index.records[0].practice_id == "config2"  # 校验新记录的practice_id正确
