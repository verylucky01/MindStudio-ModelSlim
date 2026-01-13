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


from typing import Optional, Literal, List

import torch
from pydantic import Field, ConfigDict
from torch import nn

from msmodelslim.ir.api import calculate_qparam
from msmodelslim.ir.qal import QScope, QDType, QABCRegistry
from msmodelslim.core.base.protocol import BatchProcessRequest
from msmodelslim.ir import FakeQuantActivationPerHead, FakeQuantActivationPerToken
from msmodelslim.core.observer.recall_window import RecallWindowObserver, RecallWindowObserverConfig
from msmodelslim.core.quantizer.base import QConfig
from msmodelslim.processor.base import AutoSessionProcessor, AutoProcessorConfig
from msmodelslim.utils.config_map import ConfigSet
from msmodelslim.utils.exception import UnsupportedError
from msmodelslim.utils.logging import get_logger, logger_setter
from .interface import FA3QuantAdapterInterface, FA3QuantPlaceHolder


class FA3QuantProcessorConfig(AutoProcessorConfig):
    type: Literal["fa3_quant"] = "fa3_quant"
    qconfig: Optional[QConfig] = Field(default=None, description="量化配置，默认使用INT8 per-head symmetric")
    include: List[str] = Field(default_factory=lambda: ["*"], description="包含的模块名称")
    exclude: List[str] = Field(default_factory=lambda: [], description="排除的模块名称")

    model_config = ConfigDict(extra="forbid")
    
    def __init__(self, **data):
        super().__init__(**data)
        # 如果没有提供qconfig，使用默认的INT8 per-head symmetric配置
        if self.qconfig is None:
            self.qconfig = QConfig(
                dtype=QDType.INT8,
                scope=QScope.PER_HEAD,
                symmetric=True,
                method="minmax"
            )


class _FA3PerHeadObserver(nn.Module):
    """监测器：复用 MsMinMaxObserver 的按维度统计，得到 per-head min/max。"""

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
@logger_setter(prefix="msmodelslim.processor.fa3_quant")
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
        return self.config.qconfig.scope == QScope.PER_TOKEN

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

        # 2) 将占位模块替换为监测器
        for name, submodule in request.module.named_modules(prefix=request.name):
            if isinstance(submodule, FA3QuantPlaceHolder):
                # 适配器已据 should_inject 做过滤，这里不再重复 include/exclude 判定
                observer = _FA3PerHeadObserver(ratio=submodule.get_ratio())
                self.model.set_submodule(name, observer)

    def postprocess(self, request: BatchProcessRequest) -> None:
        # 根据qconfig的scope创建对应的IR
        qconfig = self.config.qconfig
        for name, submodule in request.module.named_modules(prefix=request.name):
            if isinstance(submodule, _FA3PerHeadObserver):
                if qconfig.scope == QScope.PER_HEAD:
                    # per-head需要从observer获取统计数据
                    if submodule.min_val is None:
                        raise UnsupportedError(
                            f"FA3 quantization at {name} collected no calibration data",
                            action="Please ensure a calibration run covers this attention path before postprocess"
                        )
                    # 形状 (1, H, 1, 1) → (H,)
                    min_v = submodule.min_val.squeeze()
                    max_v = submodule.max_val.squeeze()

                    # 根据qconfig计算量化参数
                    q_param = calculate_qparam(
                        min_val=min_v,
                        max_val=max_v,
                        q_dtype=qconfig.dtype,
                        q_scope=qconfig.scope,
                        symmetric=qconfig.symmetric,
                    )
                    ir = FakeQuantActivationPerHead(q_param)
                    self.model.set_submodule(name, ir)
                # per-token不需要observer，直接创建IR
                elif qconfig.scope == QScope.PER_TOKEN:
                    # 创建空的QParam，per-token在forward中动态计算
                    from msmodelslim.ir.qal import QParam, QScheme
                    q_param = QParam(scheme=self.config.qconfig.to_scheme())
                    ir = FakeQuantActivationPerToken(q_param)
                    self.model.set_submodule(name, ir)
                
