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
import datetime
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, List

from pydantic import BaseModel, Field, ConfigDict

from msmodelslim.app.auto_tuning import TuningHistoryManagerInfra, TuningHistory
from msmodelslim.core.tune_strategy import EvaluateResult
from msmodelslim.utils.security import yaml_safe_load, yaml_safe_dump
from msmodelslim.utils.yaml_database import YamlDatabase


class YamlDatabaseHistory(BaseModel):
    history_practice_database: YamlDatabase
    history_index_file_path: Path

    model_config = ConfigDict(arbitrary_types_allowed=True)


class TuningHistoryIndexUnit(BaseModel):
    practice_id: str
    evaluation: EvaluateResult
    time: str = Field(default_factory=lambda: str(datetime.datetime.now()))


class TuningHistoryIndex(BaseModel):
    records: List[TuningHistoryIndexUnit] = Field(default_factory=list)


class YamlTuningHistoryManager(TuningHistoryManagerInfra):
    def __init__(self):
        self.database_histories: Dict[str, YamlDatabaseHistory] = {}

    def append_history(self, database: str, history: TuningHistory) -> None:
        if database not in self.database_histories:
            history_dir = Path(database)
            self.database_histories[database] = YamlDatabaseHistory(
                history_practice_database=YamlDatabase(
                    config_dir=history_dir,
                    read_only=False,
                ),
                history_index_file_path=history_dir / 'history.yaml',
            )

        database_history = self.database_histories[database]
        with _get_modifiable_history_index(database_history.history_index_file_path) as history_index:
            history_index.records.append(TuningHistoryIndexUnit(
                practice_id=history.practice.metadata.config_id,
                evaluation=history.evaluation,
            ))
        database_history.history_practice_database[history.practice.metadata.config_id] = history.practice


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
