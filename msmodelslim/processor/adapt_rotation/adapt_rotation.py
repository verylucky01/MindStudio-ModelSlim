"""
-------------------------------------------------------------------------
This file is part of the MindStudio project.
Copyright (c) 2026 Huawei Technologies Co.,Ltd.

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

from typing import Literal, Union, Any

import torch.nn as nn
from pydantic import model_validator

from msmodelslim.core.base.protocol import BatchProcessRequest
from msmodelslim.ir.qal.qregistry import QABCRegistry
from msmodelslim.processor.base import AutoProcessorConfig, AutoSessionProcessor
from msmodelslim.utils.exception import SchemaValidateError
from msmodelslim.utils.logging import get_logger

from .adapt_rotation_stage1 import AdaptRotationStage1Processor, AdaptRotationStage1ProcessorConfig
from .adapt_rotation_stage2 import AdaptRotationStage2Processor, AdaptRotationStage2ProcessorConfig


class AdaptRotationProcessorConfig(AutoProcessorConfig):
    """
    Parent config for adapt_rotation: identified by type="adapt_rotation".
    The stage field (1 or 2) determines which concrete processor runs.
    """
    type: Literal["adapt_rotation"] = "adapt_rotation"
    stage: Literal[1, 2]
    stage_config: Union[AdaptRotationStage1ProcessorConfig, AdaptRotationStage2ProcessorConfig]

    @model_validator(mode='before')
    @classmethod
    def _build_stage_config(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        stage_val = data.get("stage")
        if data.get("type") != "adapt_rotation" or stage_val not in (1, 2):
            return data
        s1 = set(AdaptRotationStage1ProcessorConfig.model_fields) - {"type"}
        s2 = set(AdaptRotationStage2ProcessorConfig.model_fields) - {"type"}
        allowed = (s1 | {"type", "stage"}) if stage_val == 1 else (s2 | {"type", "stage"})
        disallowed = set(data) - allowed
        if disallowed:
            raise SchemaValidateError(f"stage={stage_val} allows only {sorted(allowed)}; got extra: {sorted(disallowed)}")
        if stage_val == 1:
            stage_config = AdaptRotationStage1ProcessorConfig.model_validate(
                {"type": "_adapt_rotation_stage1", **{k: data[k] for k in s1 if k in data}}
            )
        else:
            stage_config = AdaptRotationStage2ProcessorConfig.model_validate(
                {"type": "_adapt_rotation_stage2", **{k: data[k] for k in s2 if k in data}}
            )
        return {"type": "adapt_rotation", "stage": stage_val, "stage_config": stage_config}

    def __getattr__(self, name: str) -> Any:
        if name in ("type", "stage", "stage_config"):
            raise AttributeError(name)
        return getattr(self.stage_config, name)


@QABCRegistry.register(dispatch_key=AdaptRotationProcessorConfig, abc_class=AutoSessionProcessor)
class AdaptRotationProcessor(AutoSessionProcessor):
    """
    Upper-level processor for adapt_rotation: identified by type="adapt_rotation".
    Dispatches to AdaptRotationStage1Processor or AdaptRotationStage2Processor
    based on the stage field in config.
    """

    def __init__(
        self,
        model: nn.Module,
        config: AdaptRotationProcessorConfig,
        adapter: object,
        **kwargs,
    ) -> None:
        super().__init__(model)
        self.config = config
        if config.stage == 1:
            self._inner = AdaptRotationStage1Processor(model, config.stage_config, adapter, **kwargs)
        else:
            self._inner = AdaptRotationStage2Processor(model, config.stage_config, adapter, **kwargs)
        get_logger().debug("AdaptRotationProcessor delegating to %s", self._inner.__class__.__name__)

    def support_distributed(self) -> bool:
        return self._inner.support_distributed()

    def is_data_free(self) -> bool:
        return self._inner.is_data_free()

    def need_kv_cache(self):
        return self._inner.need_kv_cache()

    def preprocess(self, request: BatchProcessRequest) -> None:
        self._inner.preprocess(request)

    def postprocess(self, request: BatchProcessRequest) -> None:
        self._inner.postprocess(request)

    def pre_run(self) -> None:
        self._inner.pre_run()

    def post_run(self) -> None:
        self._inner.post_run()

    def process(self, request: BatchProcessRequest) -> None:
        self._inner.process(request)
