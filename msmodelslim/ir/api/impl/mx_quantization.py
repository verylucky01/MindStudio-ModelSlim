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
#  Adapted from https://github.com/microsoft/microxcaling/blob/main/mx_ops.py

import torch

from msmodelslim.ir.qal import QStorage
from msmodelslim.ir.qal.qbase import QDType, QScope, QParam, QScheme
from msmodelslim.ir.qal.qregistry import QFuncRegistry

FP32_EXPONENT_BIAS = 127
FP32_MIN_NORMAL = 2 ** (-FP32_EXPONENT_BIAS + 1)


@QFuncRegistry.register(dispatch_key=(QDType.MXFP8, QScope.PER_BLOCK, True), api_name="calculate_qparam")
@QFuncRegistry.register(dispatch_key=(QDType.MXFP4, QScope.PER_BLOCK, True), api_name="calculate_qparam")
def calculate_mx_qparam(
        min_val: torch.Tensor,
        max_val: torch.Tensor,
        q_dtype: QDType,
        q_scope: QScope,
        symmetric: bool,
        **kwargs
) -> QParam:
    mx_finfo = q_dtype.mx_finfo
    is_flush_fp32_subnorms = mx_finfo.flush_fp32_subnorms

    shared_exp = torch.floor(
        torch.log2(max_val + FP32_MIN_NORMAL * (max_val == 0).to(max_val.dtype))
    )
    shared_exp = shared_exp - mx_finfo.emax

    if is_flush_fp32_subnorms:
        # 标记需要保留的 shared_exp (bool mask)，调用方可据此清零 A
        keep_mask = (shared_exp > -FP32_EXPONENT_BIAS)
    else:
        keep_mask = None


    scale_emax = 2 ** (mx_finfo.scale_bits - 1) - 1
    shared_exp[shared_exp > scale_emax] = float("NaN")
    shared_exp[shared_exp < -scale_emax] = -scale_emax

    return QParam(
        scheme=QScheme(
            dtype=q_dtype,
            scope=q_scope,
            symmetric=symmetric,
        ),
        ext={
            "scale": shared_exp,
            "offset": torch.zeros_like(shared_exp),
            "keep_mask": keep_mask
        }
    )


@QFuncRegistry.register(dispatch_key=(QDType.FLOAT, QDType.MXFP8, QScope.PER_BLOCK, True), api_name="quantize")
@QFuncRegistry.register(dispatch_key=(QDType.FLOAT, QDType.MXFP4, QScope.PER_BLOCK, True), api_name="quantize")
def mxfp_per_block_quantize(tensor: QStorage, q_param: QParam) -> QStorage:
    mx_finfo = q_param.scheme.dtype.mx_finfo
    inp = tensor.value
    dtype = inp.dtype

    inp = inp.to(torch.float32)
    shared_exp = q_param.ext['scale']
    keep_mask = q_param.ext.get('keep_mask', None)

    if keep_mask is not None:
        inp = inp * keep_mask.to(inp.dtype)

    inp = inp / (2 ** shared_exp)
    private_exp = torch.floor(torch.log2(torch.abs(inp) + (inp == 0).to(inp.dtype)))

    inp_ = inp.clone()
    inp = _quant(inp, mx_finfo.mbits, private_exp, mx_finfo.ebits)
    inp = _clamp_out(inp, inp_, mx_finfo.max_norm)
    inp = inp.to(dtype)
    del inp_

    tensor_q = tensor.same_like(inp).to(q_param.scheme.dtype)
    return tensor_q


@QFuncRegistry.register(dispatch_key=(QDType.MXFP8, QDType.MXFP8, QScope.PER_BLOCK, True), api_name="dequantize")
@QFuncRegistry.register(dispatch_key=(QDType.MXFP4, QDType.MXFP4, QScope.PER_BLOCK, True), api_name="dequantize")
def mxfp_per_block_dequantize(tensor: QStorage, q_param: QParam) -> QStorage:
    shared_exp = q_param.ext['scale']
    quant_inp = tensor.value
    dtype = quant_inp.dtype
    inp = quant_inp * (2 ** shared_exp)
    tensor_q = tensor.same_like(inp).to(dtype)
    return tensor_q


def _quant(a, bits, exp, exp_bits):
    # +2 的偏移是为了计算 max_norm ，此处需要进行加减
    min_exp = - (2 ** (exp_bits - 1)) + 2
    exp = exp.clip(min=min_exp)
    bits_ = bits - 2

    if exp is None: # 私有指数为空
        a = a * (2 ** bits_)
        a = torch.sign(a) * torch.floor(torch.abs(a) + 0.5)
        a = a / (2 ** bits_)
    else:
        a = a / (2 ** exp) * (2 ** bits_)
        a = torch.sign(a) * torch.floor(torch.abs(a) + 0.5)
        a = a / (2 ** bits_) * (2 ** exp)
    return a


def _clamp_out(out, a, max_norm):
    out = torch.clamp(out, min=-max_norm, max=max_norm)
    out[a == float("Inf")] = float("Inf")
    out[a == -float("Inf")] = -float("Inf")
    out[a == float("NaN")] = float("NaN")
    return out

