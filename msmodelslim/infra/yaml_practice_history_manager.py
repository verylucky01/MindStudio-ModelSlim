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
EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
See the Mulan PSL v2 for more details.
-------------------------------------------------------------------------
"""
import datetime
import hashlib
import json
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, List, Optional

from pydantic import BaseModel, Field, ConfigDict

from msmodelslim.app.auto_tuning import TuningHistoryManagerInfra, TuningHistoryInfra
from msmodelslim.core.practice import PracticeConfig
from msmodelslim.core.tune_strategy import EvaluateResult, EvaluateAccuracy
from msmodelslim.utils.logging import get_logger
from msmodelslim.utils.security import (
    yaml_safe_load,
    yaml_safe_dump,
    safe_delete_path_if_exists,
    get_write_directory,
)
from msmodelslim.utils.yaml_database import YamlDatabase


class YamlDatabaseHistory(BaseModel):
    history_practice_database: YamlDatabase  # Record database: manages practice config files
    history_index_file_path: Path  # History index: manages history.yaml
    accuracy_database: YamlDatabase  # Accuracy database: manages accuracy.yaml

    model_config = ConfigDict(arbitrary_types_allowed=True)


class TuningHistoryIndexUnit(BaseModel):
    practice_id: str
    evaluation: EvaluateResult
    md5: str
    time: str = Field(default_factory=lambda: str(datetime.datetime.now()))


class TuningHistoryIndex(BaseModel):
    records: List[TuningHistoryIndexUnit] = Field(default_factory=list)


class YamlTuningHistory(TuningHistoryInfra):
    """
    YAML-based implementation of TuningHistoryInfra.
    Manages history.yaml (history index) and accuracy.yaml (accuracy cache).
    """
    
    def __init__(self, database: str):
        """
        Initialize YamlTuningHistory.
        
        Args:
            database: str, the path to the history database directory
        """
        self.database = database
        get_write_directory(database)
        self.history_dir = Path(database)
        
        # Load or create history index
        history_index_file_path = self.history_dir / 'history.yaml'
        
        # Initialize database
        accuracy_database_dir = self.history_dir
        self._database_history = YamlDatabaseHistory(
            history_practice_database=YamlDatabase(
                config_dir=self.history_dir,
                read_only=False,
            ),
            history_index_file_path=history_index_file_path,
            accuracy_database=YamlDatabase(
                config_dir=accuracy_database_dir,
                read_only=False,
            ),
        )
        
        # Always start with new history index
        self._history_index = TuningHistoryIndex()
        
        # Load accuracy cache into memory
        self._accuracy_cache: Dict[str, Dict] = self._load_accuracy_database()
    
    def clear_records(self) -> None:
        """
        Clear history records (history.yaml and practice config files), but preserve accuracy cache.
        """
        get_logger().info("Clearing history records (preserving accuracy cache)...")
        
        # Clear practice config files
        record_database = self._database_history.history_practice_database
        excluded_files = {'accuracy', 'history'}
        practice_ids_to_delete = [pid for pid in record_database if pid not in excluded_files]
        for practice_id in practice_ids_to_delete:
            practice_file = record_database.config_dir / f"{practice_id}.yaml"
            if practice_file.exists():
                safe_delete_path_if_exists(str(practice_file), logger_level="info")
        
        # Clear history index
        with _get_modifiable_history_index(self._database_history.history_index_file_path) as history_index:
            history_index.records.clear()
            self._history_index = history_index
        
        get_logger().info("History records cleared successfully")
        
    def _load_accuracy_database(self) -> Dict[str, Dict]:
        """Load accuracy data from accuracy database interface."""
        accuracy_database = self._database_history.accuracy_database
        accuracy_key = "accuracy"
        
        if accuracy_key not in accuracy_database:
            get_logger().debug("Accuracy database key %r does not exist. Starting with empty cache.", accuracy_key)
            return {}
        
        try:
            content = accuracy_database[accuracy_key]
            if content is None:
                return {}
            
            return content if isinstance(content, dict) else {}
        except Exception as e:
            get_logger().debug("Failed to load accuracy cache from database: %s. Starting with empty cache.", e)
            return {}
    
    def get_accuracy(self, practice: PracticeConfig) -> Optional[EvaluateResult]:
        """Get accuracy from history for the given practice by MD5."""
        practice_md5 = calculate_practice_md5(practice)
        
        if practice_md5 not in self._accuracy_cache:
            return None
        
        evaluation_dict = self._accuracy_cache[practice_md5]
        get_logger().info("Found accuracy in history. Using historical accuracy.")
        return EvaluateResult.model_validate(evaluation_dict)
    
    def append_history(self, practice: PracticeConfig, evaluation: EvaluateResult) -> None:
        """
        Append a history record to the database.
        This method will also save accuracy data to the accuracy cache.
        """
        practice_id = practice.metadata.config_id
        practice_md5 = calculate_practice_md5(practice)
        
        if practice_md5 in self._accuracy_cache:
            get_logger().warning(
                "Overwriting existing accuracy data for practice MD5 %r. "
                "Previous accuracy data will be replaced with new evaluation result.",
                practice_md5
            )
        
        evaluation_dict = evaluation.model_dump()
        self._accuracy_cache[practice_md5] = evaluation_dict
        self._database_history.accuracy_database["accuracy"] = self._accuracy_cache
        
        # Append to history index
        with _get_modifiable_history_index(self._database_history.history_index_file_path) as history_index:
            history_index.records.append(TuningHistoryIndexUnit(
                practice_id=practice_id,
                evaluation=evaluation,
                md5=practice_md5,
            ))
            self._history_index = history_index
        
        # Save practice config (this will replace if exists)
        self._database_history.history_practice_database[practice_id] = practice
    
    def get_accuracy_count(self) -> int:
        """Return the number of accuracy records in the accuracy cache."""
        return len(self._accuracy_cache)


class YamlTuningHistoryManager(TuningHistoryManagerInfra):
    """Manager for loading YAML-based tuning history."""
    
    def load_history(self, database: str) -> TuningHistoryInfra:
        """
        Load the complete tuning history from the specified database path.
        Returns a history instance even if no records exist (empty history).
        
        Args:
            database: str, the path to the history database directory
            
        Returns:
            TuningHistoryInfra: The tuning history instance
            
        Raises:
            RuntimeError: If failed to create or load history
        """
        try:
            return YamlTuningHistory(database)
        except Exception as e:
            error_msg = f"Failed to create or load history from {database}. Cannot proceed with tuning."
            get_logger().error(error_msg)
            raise RuntimeError(error_msg) from e


def calculate_practice_md5(practice: PracticeConfig) -> str:
    """
    Calculate MD5 hash of a practice object.
    
    Args:
        practice: PracticeConfig object
        
    Returns:
        str: MD5 hash string
    """
    practice_dict = practice.model_dump()
    practice_json = json.dumps(practice_dict, sort_keys=True)
    return hashlib.md5(practice_json.encode('utf-8')).hexdigest()


@contextmanager
def _get_modifiable_history_index(history_index_file_path: Path):
    """
    作为上下文管理器，负责：
    1. 从索引文件中加载 PracticeHistoryIndex
    2. 将修改后的 PracticeHistoryIndex 回写到索引文件
    """
    if history_index_file_path.exists():
        history_content = yaml_safe_load(str(history_index_file_path))
        history_index = TuningHistoryIndex.model_validate(history_content)
    else:
        history_index = TuningHistoryIndex()
    try:
        yield history_index
    finally:
        yaml_safe_dump(history_index.model_dump(), str(history_index_file_path))
