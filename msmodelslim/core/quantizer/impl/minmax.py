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

from typing import Optional

import torch
from pydantic import validate_call

import msmodelslim.ir as qir
from msmodelslim.ir.api import fake_quantize, quantize, dequantize, calculate_qparam
from msmodelslim.ir.qal import QABCRegistry, QDType, QStorage, QParam, QScope, QScheme
from msmodelslim.core.observer import MsMinMaxObserver, MinMaxObserverConfig
from msmodelslim.utils.exception import SpecError, SchemaValidateError
from msmodelslim.utils.logging import logger_setter
from msmodelslim.core.observer import MsMinMaxBlockObserver, MinMaxBlockObserverConfig
from msmodelslim.ir.utils import reshape_to_blocks, undo_reshape_to_blocks
from ..base import AutoActQuantizer, AutoWeightQuantizer, QConfig


@QABCRegistry.multi_register(
    dispatch_key=[
        (qir.int8_per_tensor_sym, "minmax"),
        (qir.int8_per_tensor_asym, "minmax"),
        (qir.fp8_e4m3_per_tensor_sym, "minmax"),
    ],
    abc_type=AutoActQuantizer
)
@logger_setter()
class ActPerTensorMinmax(AutoActQuantizer):

    def __init__(self, config: QConfig):
        super().__init__()
        self.config = config
        minmax_config = MinMaxObserverConfig.model_validate({})
        self.minmax_observer = MsMinMaxObserver(minmax_config)
        self.q_param: Optional[QParam] = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.minmax_observer.update(x, self.sync)
        min_val, max_val = self.minmax_observer.get_min_max()
        self.q_param = calculate_qparam(
            min_val=min_val,
            max_val=max_val,
            q_dtype=QDType(self.config.dtype),
            q_scope=QScope(self.config.scope),
            symmetric=self.config.symmetric,
        )
        return fake_quantize(QStorage(dtype=QDType.FLOAT, value=x), self.q_param).value

    def get_q_param(self) -> QParam:
        if self.q_param is None:
            raise SpecError("No q_param was set", action="Please call forward first")
        return self.q_param


@QABCRegistry.multi_register(
    dispatch_key=[
        (qir.int8_per_token_asym, "minmax"),
        (qir.int8_per_token_sym, "minmax"),
        (qir.fp8_e4m3_per_token_sym, "minmax"),
    ],
    abc_type=AutoActQuantizer
)
@logger_setter()
class ActPerTokenMinmax(AutoActQuantizer):

    def __init__(self, config: QConfig):
        super().__init__()
        self.config = config
        minmax_config = MinMaxObserverConfig.model_validate({})
        self.minmax_observer = MsMinMaxObserver(minmax_config)
        self.q_param: Optional[QParam] = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_shape = x.shape
        x_reshaped = x.reshape(-1, x.shape[-1])
        self.minmax_observer.reset()
        self.minmax_observer.update(x_reshaped, self.sync)
        min_val, max_val = self.minmax_observer.get_min_max()
        self.q_param = calculate_qparam(
            min_val=min_val,
            max_val=max_val,
            q_dtype=QDType(self.config.dtype),
            q_scope=QScope(self.config.scope),
            symmetric=self.config.symmetric,
        )
        return fake_quantize(QStorage(dtype=QDType.FLOAT, value=x), self.q_param).value.reshape(x_shape)

    def get_q_param(self) -> QParam:
        if self.q_param is None:
            return QParam(scheme=self.config.to_scheme())
        return self.q_param


@QABCRegistry.multi_register(
    dispatch_key=[
        (qir.int8_per_channel_sym, "minmax"),
        (qir.int8_per_channel_asym, "minmax"),
        (qir.fp8_e4m3_per_channel_sym, "minmax"),
    ],
    abc_type=AutoActQuantizer
)
class ActPerChannelMinmax(AutoActQuantizer):
    def __init__(self, config: QConfig):
        super().__init__()
        self.config = config
        minmax_config = MinMaxObserverConfig(dim=0, keepdim=False)
        self.minmax_observer = MsMinMaxObserver(minmax_config)
        self.q_param: Optional[QParam] = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_shape = x.shape
        x_reshaped = x.reshape(-1, x.shape[-1])
        self.minmax_observer.reset()
        self.minmax_observer.update(x_reshaped, self.sync)
        min_val, max_val = self.minmax_observer.get_min_max()
        self.q_param = calculate_qparam(
            min_val=min_val,
            max_val=max_val,
            q_dtype=QDType(self.config.dtype),
            q_scope=QScope(self.config.scope),
            symmetric=self.config.symmetric,
        )
        return fake_quantize(QStorage(dtype=QDType.FLOAT, value=x), self.q_param).value.reshape(x_shape)

    def get_q_param(self) -> QParam:
        if self.q_param is None:
            raise RuntimeError("No q_param was set, please call forward first")
        return self.q_param


@QABCRegistry.multi_register(
    dispatch_key=[
        (qir.int8_pd_mix_asym, "minmax")
    ],
    abc_type=AutoActQuantizer
)
@logger_setter()
class ActPDMixMinmax(AutoActQuantizer):

    def __init__(self, config: QConfig):
        super().__init__()
        self.config = config
        minmax_config = MinMaxObserverConfig.model_validate({})
        self.prefilling_observer = MsMinMaxObserver(minmax_config)
        self.decoding_observer = MsMinMaxObserver(minmax_config)
        self.q_param: Optional[QParam] = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # calculate decoding static param
        self.decoding_observer.update(x, self.sync)
        min_val, max_val = self.decoding_observer.get_min_max()
        self.q_param = calculate_qparam(
            min_val=min_val,
            max_val=max_val,
            q_dtype=QDType(self.config.dtype),
            q_scope=QScope.PER_TENSOR,
            symmetric=False,
        )

        # calibration as prefilling as dynamic
        x_shape = x.shape
        x_reshaped = x.reshape(-1, x.shape[-1])
        self.prefilling_observer.reset()
        self.prefilling_observer.update(x_reshaped, self.sync)
        min_val, max_val = self.prefilling_observer.get_min_max()
        tmp_q_param = calculate_qparam(
            min_val=min_val,
            max_val=max_val,
            q_dtype=QDType(self.config.dtype),
            q_scope=QScope.PER_TOKEN,
            symmetric=True,
        )
        return fake_quantize(QStorage(dtype=QDType.FLOAT, value=x), tmp_q_param).value.reshape(x_shape)

    def get_q_param(self) -> QParam:
        if self.q_param is None:
            raise SpecError("No q_param was set", action="Please call forward first")
        return QParam(scheme=qir.int8_pd_mix_asym, ext=self.q_param.ext)


@QABCRegistry.multi_register(
    dispatch_key=[
        (qir.int8_per_channel_sym, "minmax"),
        (qir.int4_per_channel_sym, "minmax"),
        (qir.int4_per_channel_sym, "minmax"),
        (qir.fp8_e4m3_per_channel_sym, "minmax"),
    ],
    abc_type=AutoWeightQuantizer
)
@logger_setter()
class WeightPerChannelMinmax(AutoWeightQuantizer):
    def __init__(self, config: QConfig):
        super().__init__()
        minmax_config = MinMaxObserverConfig(dim=0, keepdim=False)
        self.config = config
        self.minmax_observer = MsMinMaxObserver(minmax_config)
        self.weight: Optional[QStorage] = None
        self.bias: Optional[torch.Tensor] = None
        self.w_q_param: Optional[QParam] = None
        self.w_q_storage: Optional[QStorage] = None

    def forward(self, x: Optional[torch.Tensor] = None) -> torch.Tensor:
        if self.weight is None:
            raise SpecError("No weight was set", action="please call init_weight first")
        self.minmax_observer.update(self.weight.T.value, self.sync)
        min_val, max_val = self.minmax_observer.get_min_max()
        self.w_q_param = calculate_qparam(
            min_val=min_val,
            max_val=max_val,
            q_dtype=QDType(self.config.dtype),
            q_scope=QScope(self.config.scope),
            symmetric=self.config.symmetric,
        )
        self.w_q_storage = quantize(self.weight.T, self.w_q_param).T
        return dequantize(self.w_q_storage.T, self.w_q_param).T.value

    @validate_call(config=dict(arbitrary_types_allowed=True))
    def init_weight(self, weight: QStorage, bias: Optional[torch.Tensor] = None) -> None:
        self.weight = weight
        self.bias = bias

    def get_q_storage(self) -> QStorage:
        if self.w_q_storage is None:
            _ = self.forward(None)
        return self.w_q_storage

    def get_q_param(self) -> QParam:
        if self.w_q_param is None:
            _ = self.forward(None)
        return self.w_q_param


@QABCRegistry.multi_register(
    dispatch_key=[
        (qir.mxfp8_per_block_sym, "minmax"),
        (qir.mxfp4_per_block_sym, "minmax"),
    ],
    abc_type=AutoWeightQuantizer
)
@logger_setter()
class MXWeightPerBlockMinmax(AutoWeightQuantizer):
    def __init__(self, config: QConfig):
        super().__init__()

        self.config = config

        self.axes = config.ext.get('axes', -1)
        if not isinstance(self.axes, (int, list)):
            raise SchemaValidateError(
                f"Invalid value for 'axes': {self.axes}. Expected int or list[int]."
            )
        self.block_size = config.dtype.mx_finfo.block_size

        self.w_q_param: Optional[QParam] = None
        self.w_q_storage: Optional[QStorage] = None

    def forward(self, x: Optional[torch.Tensor] = None) -> torch.Tensor:
        if self.weight is None:
            raise SpecError("No weight was set", action="please call init_weight first")

        dequant_value = dequantize(self.w_q_storage, self.w_q_param).value
        dequant_value = undo_reshape_to_blocks(dequant_value, padded_shape, orig_shape, axes)
        return dequant_value

    @validate_call(config=dict(arbitrary_types_allowed=True))
    def init_weight(self, weight: QStorage, bias: Optional[torch.Tensor] = None) -> None:
        minmax_config = MinMaxBlockObserverConfig(axes=self.axes)
        minmax_block_observer = MsMinMaxBlockObserver(minmax_config)
        weight_value = weight.value.detach()
        axes = self.axes
        axes = [axes] if isinstance(axes, int) else axes
        axes = [x + weight_value.ndim if x < 0 else x for x in axes]

        weight_value, axes_, orig_shape, padded_shape = reshape_to_blocks(weight_value, axes, self.block_size)
        shared_exp_axes = [x + 1 for x in axes_] if self.block_size > 0 else axes_

        minmax_block_observer.update(weight_value, sync=self.sync, shared_exp_axes=shared_exp_axes)

        min_val, max_val = minmax_block_observer.get_min_max()

        self.w_q_param = calculate_qparam(
            min_val=min_val,
            max_val=max_val,
            q_dtype=QDType(self.config.dtype),
            q_scope=QScope(self.config.scope),
            symmetric=self.config.symmetric,
            axes=self.config.ext.get('axes')
        )
        self.w_q_param.ext['axes'] = self.axes
        self.w_q_storage = quantize(QStorage(QDType.FLOAT, weight_value), self.w_q_param)
        dequant_value = dequantize(self.w_q_storage, self.w_q_param).value
        dequant_value = undo_reshape_to_blocks(dequant_value, padded_shape, orig_shape, axes)
        self.w_q_storage.value = undo_reshape_to_blocks(self.w_q_storage.value, padded_shape, orig_shape, axes)

    def get_q_storage(self) -> QStorage:
        return self.w_q_storage

    def get_q_param(self) -> QParam:
        return self.w_q_param


@QABCRegistry.multi_register(
    dispatch_key=[
        (qir.mxfp8_per_block_sym, "minmax"),
        (qir.mxfp4_per_block_sym, "minmax"),
    ],
    abc_type=AutoActQuantizer
)
@logger_setter()
class MXActPerBlockMinmax(AutoActQuantizer):

    def __init__(self, config: QConfig):
        super().__init__()
        self.config = config
        self.axes = config.ext.get('axes', -1)
        if not isinstance(self.axes, (int, list)):
            raise SchemaValidateError(
                f"Invalid value for 'axes': {self.axes}. Expected int or list[int]."
            )
        self.q_param = QParam(
            scheme=QScheme(
                dtype=config.dtype,
                scope=config.scope,
                symmetric=config.symmetric,
            ),
            ext={
                'axes': self.axes
            }
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x

    def get_q_param(self) -> QParam:
        if self.q_param is None:
            return QParam(scheme=self.config.to_scheme())
        return self.q_param