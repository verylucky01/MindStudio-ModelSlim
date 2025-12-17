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

from typing import Optional, Literal, List

import torch
from pydantic import Field, ConfigDict
from torch import nn

from msmodelslim.core import calculate_qparam, QDType
from msmodelslim.core.QAL import QScope
from msmodelslim.core.QAL.qregistry import QABCRegistry
from msmodelslim.core.base.protocol import BatchProcessRequest
from msmodelslim.quant.ir import FakeQuantActivationPerHead
from msmodelslim.quant.observer.recall_window import RecallWindowObserver, RecallWindowObserverConfig
from msmodelslim.quant.processor.base import AutoSessionProcessor, AutoProcessorConfig
from msmodelslim.utils.config_map import ConfigSet
from msmodelslim.utils.exception import UnsupportedError
from msmodelslim.utils.logging import get_logger, logger_setter
from .interface import FA3QuantAdapterInterface, FA3QuantPlaceHolder


class FA3QuantProcessorConfig(AutoProcessorConfig):
    type: Literal["fa3_quant"] = "fa3_quant"
    include: List[str] = Field(default_factory=lambda: ["*"], description="包含的模块名称")
    exclude: List[str] = Field(default_factory=lambda: [], description="排除的模块名称")

    model_config = ConfigDict(extra="forbid")


class _FA3PerHeadObserver(nn.Module):
    """监听器：复用 MsMinMaxObserver 的按维度统计，得到 per-head min/max。"""

    def __init__(self, ratio: float = 1.0):
        super().__init__()
        self._observer = RecallWindowObserver(
            RecallWindowObserverConfig(
                ratio=ratio,
                dim=-1,
                keepdim=True))

    @property
    def min_val(self) -> Optional[torch.Tensor]:
        return self._observer.get_min()

    @property
    def max_val(self) -> Optional[torch.Tensor]:
        return self._observer.get_max()
    
    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 对 (B, H, S, D) 按 [0, 2, 3] 归约，保留 H 维；keepdim=True 得到形如 (1, H, 1, 1)
        samples = x.contiguous().view(x.shape[1], -1)
        self._observer.update(samples)
        return x


@QABCRegistry.register(dispatch_key=FA3QuantProcessorConfig, abc_class=AutoSessionProcessor)
@logger_setter("msmodelslim.quant.processor.quant.fa3")
class FA3QuantProcessor(AutoSessionProcessor):
    def __init__(
            self,
            model: nn.Module,
            config: FA3QuantProcessorConfig,
            adapter: Optional[object] = None,
    ):
        super().__init__(model)
        self.config = config
        if not isinstance(adapter, FA3QuantAdapterInterface):
            raise UnsupportedError(
                f"Adapter {adapter.__class__.__name__} does not implement FA3QuantAdapterInterface",
                action="Please implement FA3QuantAdapterInterface"
            )
        self.adapter = adapter
        self.include = ConfigSet(config.include)
        self.exclude = ConfigSet(config.exclude)

    def is_data_free(self) -> bool:
        return False

    def support_distributed(self) -> bool:
        return False

    def preprocess(self, request: BatchProcessRequest) -> None:
        # 1) 调用适配器接口注入占位模块（如果提供）
        # 期望适配器实现方法：install_fa3_placeholders(module, should_inject) -> None
        try:
            self.adapter.inject_fa3_placeholders(
                request.name,
                request.module,
                lambda module_name: (module_name in self.include and module_name not in self.exclude)
            )
        except Exception as e:
            get_logger().warning(f"install fa3 placeholders at {request.name} failed: {e}")

        # 2) 将占位模块替换为监听器
        for name, submodule in request.module.named_modules(prefix=request.name):
            if isinstance(submodule, FA3QuantPlaceHolder):
                # 适配器已据 should_inject 做过滤，这里不再重复 include/exclude 判定
                observer = _FA3PerHeadObserver(ratio=submodule.get_ratio())
                self.model.set_submodule(name, observer)

    def postprocess(self, request: BatchProcessRequest) -> None:
        # 汇总监听器数据，计算 per-head 对称 scale，并替换为 IR
        for name, submodule in request.module.named_modules(prefix=request.name):
            if isinstance(submodule, _FA3PerHeadObserver):
                if submodule.min_val is None:
                    raise UnsupportedError(
                        f"FA3 quantization at {name} collected no calibration data",
                        action="Please ensure a calibration run covers this attention path before postprocess"
                    )
                # 形状 (1, H, 1, 1) → (H,)
                min_v = submodule.min_val.squeeze()
                max_v = submodule.max_val.squeeze()

                # 固定 per-head 对称 INT8 方案
                q_param = calculate_qparam(
                    min_val=min_v,
                    max_val=max_v,
                    q_dtype=QDType.INT8,
                    q_scope=QScope.PER_HEAD,
                    symmetric=True,
                )
                ir = FakeQuantActivationPerHead(q_param)
                self.model.set_submodule(name, ir)
