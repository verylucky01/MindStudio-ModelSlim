# Copyright Huawei Technologies Co., Ltd. 2025. All rights reserved.
from abc import ABC
from pathlib import Path
from typing import Dict, Generator, Optional

from msmodelslim.app.naive_quantization import PracticeManagerInfra as nqpm
from msmodelslim.app.practice.interface import PracticeConfig
from msmodelslim.utils.exception import SecurityError, UnsupportedError
from msmodelslim.utils.security import get_valid_read_path
from msmodelslim.utils.yaml_database import YamlDatabase


class PracticeManager(nqpm, ABC):
    def __init__(self, official_config_dir: Path, custom_config_dir: Optional[Path] = None):
        get_valid_read_path(str(official_config_dir), is_dir=True)
        self.official_config_dir = official_config_dir
        if custom_config_dir is not None:
            get_valid_read_path(str(custom_config_dir), is_dir=True)
        self.custom_config_dir = custom_config_dir

        self.official_databases: Dict[str, YamlDatabase] = {
            model_type_dir.name: YamlDatabase(model_type_dir, read_only=True)
            for model_type_dir in self.official_config_dir.iterdir()
            if model_type_dir.is_dir()
        }

        self.custom_databases: Dict[str, YamlDatabase] = {
            model_type_dir.name: YamlDatabase(model_type_dir, read_only=False)
            for model_type_dir in self.custom_config_dir.iterdir()
            if model_type_dir.is_dir()
        } if self.custom_config_dir else {}

    def __contains__(self, model_pedigree: str) -> bool:
        model_pedigree = model_pedigree.lower()
        return model_pedigree in self.custom_databases or model_pedigree in self.official_databases

    def get_config_by_id(self, model_pedigree: str, config_id: str) -> PracticeConfig:
        model_pedigree = model_pedigree.lower()
        if model_pedigree in self.custom_databases and config_id in self.custom_databases[model_pedigree]:
            value = self.custom_databases[model_pedigree][config_id]
        elif model_pedigree in self.official_databases and config_id in self.official_databases[model_pedigree]:
            value = self.official_databases[model_pedigree][config_id]
        else:
            raise UnsupportedError(f"Practice {config_id} of ModelType {model_pedigree} not found",
                                   action='Please check the practice id and model type')

        quant_config = PracticeConfig.from_dict(value)

        if config_id != quant_config.metadata.config_id:
            raise SecurityError(f"name {config_id} not match config_id {quant_config.metadata.config_id}",
                                action='Please make sure the practice is not tampered')
        return quant_config

    def iter_config(self, model_pedigree) -> Generator[PracticeConfig, None, None]:
        tasks = []
        if model_pedigree in self.custom_databases:
            for value in self.custom_databases[model_pedigree].values():
                tasks.append(PracticeConfig.from_dict(value))
        if model_pedigree in self.official_databases:
            for value in self.official_databases[model_pedigree].values():
                tasks.append(PracticeConfig.from_dict(value))

        if not tasks:
            raise UnsupportedError(f"Model type {model_pedigree} not found in practice repository",
                                   action='Please check the model type')

        tasks.sort(key=lambda x: (-x.metadata.score, x.metadata.config_id))
        for task in tasks:
            yield task
