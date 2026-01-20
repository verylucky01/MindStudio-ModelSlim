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
from pathlib import Path
from typing import Literal, List, Annotated

from pydantic import AfterValidator, BaseModel

from msmodelslim.app.auto_tuning import EvaluateServiceInfra, EvaluateServiceConfig
from msmodelslim.app.auto_tuning.evaluation_service_infra import EvaluateContext
from msmodelslim.core.tune_strategy import EvaluateResult, EvaluateAccuracy, AccuracyExpectation
from msmodelslim.infra.evaluation.aisbench_server import AisBenchServer, AisbenchServerConfig
from msmodelslim.infra.vllm_ascend_server import VllmAscendServer, VllmAscendConfig
from msmodelslim.utils.exception import SpecError
from msmodelslim.utils.logging import logger_setter
from msmodelslim.utils.plugin import TypedConfig
from msmodelslim.utils.validation.pydantic import at_least_one_element


class EvaluateDemand(BaseModel):
    expectations: Annotated[List[AccuracyExpectation], AfterValidator(at_least_one_element)]


class ServiceOrientedEvaluateServiceConfig(EvaluateServiceConfig):
    type: TypedConfig.TypeField = Literal['service_oriented']
    demand: EvaluateDemand
    evaluation: AisbenchServerConfig
    inference_engine: VllmAscendConfig


@logger_setter()
class ServiceOrientedEvaluateService(EvaluateServiceInfra):
    def evaluate(self,
                 context: EvaluateContext,
                 evaluate_config: ServiceOrientedEvaluateServiceConfig,
                 model_path: Path,
                 ) -> EvaluateResult:
        server = None
        try:
            server = VllmAscendServer(
                context=context,
                model_path=model_path,
                server_config=evaluate_config.inference_engine,
                log_file_path=context.working_dir / "vllm_server.log"
            )

            if not server.start():
                raise SpecError("[ServiceOrientedEvaluateService] VLLM failed to start")

            bencher = AisBenchServer(
                context=context,
                eval_config=evaluate_config.evaluation,
                datasets=[d.dataset for d in evaluate_config.demand.expectations],
                quantized_model_path=model_path,
                current_run_dir=context.working_dir,
            )
            accuracies = bencher.run()
            return EvaluateResult(
                accuracies=accuracies,
                expectations=evaluate_config.demand.expectations,
                is_satisfied=is_demand_satisfied(
                    demand=evaluate_config.demand.expectations,
                    evaluate_result=accuracies,
                ),
            )
        finally:
            if server and server.process.process:
                server.stop()


def is_demand_satisfied(
        demand: List[AccuracyExpectation],
        evaluate_result: List[EvaluateAccuracy],
) -> bool:
    """判断 result 是否覆盖并满足所有 demand 的精度要求。"""

    demand_datasets = [d.dataset for d in demand]
    if len(demand_datasets) != len(set(demand_datasets)):
        raise SpecError("Duplicate dataset found in demand.")

    # 使用 dict 同时检测重复和构建索引
    result_map = {r.dataset: r for r in evaluate_result}
    if len(result_map) != len(evaluate_result):
        raise SpecError("Duplicate dataset found in result.")

    # result 至少要覆盖所有 demand 的 dataset
    if not set(demand_datasets).issubset(result_map.keys()):
        return False

    for d in demand:
        r = result_map[d.dataset]
        if d.target - r.accuracy > d.tolerance:
            return False

    return True
