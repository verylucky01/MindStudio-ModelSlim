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
from pathlib import Path

from msmodelslim.app.auto_tuning import TuningPlanManagerInfra, TuningPlanConfig
from msmodelslim.utils.exception import SpecError
from msmodelslim.utils.security import yaml_safe_load
from msmodelslim.utils.yaml_database import YamlDatabase


class YamlTuningPlanManager(TuningPlanManagerInfra):
    def get_plan_by_id(self, plan_id: str) -> TuningPlanConfig:
        content = yaml_safe_load(plan_id)
        return TuningPlanConfig.model_validate(content)
