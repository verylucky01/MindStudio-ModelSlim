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

import os
from typing import Dict, Any, Optional, List, Literal

import torch
import torch.distributed as dist
from pydantic import Field
from torch import nn

from ascend_utils.common.security.path import json_safe_load, json_safe_dump
from msmodelslim import logger, ir as qir
from msmodelslim.ir.qal.qregistry import QABCRegistry
from msmodelslim.model import IModel
from msmodelslim.processor.base import AutoSessionProcessor
from msmodelslim.utils.distributed import DistHelper
from msmodelslim.utils.exception import ToDoError, SchemaValidateError
from msmodelslim.utils.security import safe_copy_file
from msmodelslim.utils.logging import logger_setter
from .interface import AscendV1SaveInterface
from .saver import AutoSaverProcessor, AutoSaverBaseConfig
from .utils.json import JsonWriter
from .utils.pack import w4a8_pack_int4, process_scale
from .utils.safetensors import SafetensorsWriter, BufferedSafetensorsWriter


def copy_files(input_path, output_path):
    """
    复制模型配置文件
    @param input_path: 源目录
    @param output_path: 目标目录
    """
    for file in os.listdir(input_path):
        if not any((file.endswith(subfix) for subfix in ['.json', '.py'])):
            continue

        if any((file.endswith(subfix) for subfix in ['index.json'])):
            continue

        ori_file = os.path.join(input_path, file)
        dest_file = os.path.join(output_path, file)
        safe_copy_file(src_path=ori_file, dest_path=dest_file)
        os.chmod(dest_file, int("600", 8))


def remove_quantization_config(output_path):
    """
    从config.json文件中移除quantization_config字段
    @param output_path: 目标目录
    """
    config_file = os.path.join(output_path, "config.json")

    if not os.path.exists(config_file):
        return

    try:
        config_data = json_safe_load(config_file, check_user_stat=True)
        
        if 'quantization_config' in config_data:
            del config_data['quantization_config']
            json_safe_dump(config_data, config_file, indent=2, check_user_stat=True)
    except Exception as e:
        logger.warning(f"Failed to remove quantization_config in config.json!")


class AscendV1Config(AutoSaverBaseConfig):
    """
    ascendV1 量化模型保存器配置。该配置用于配置ascendV1量化模型保存器。
    
    该配置包含以下字段：
        - type: 量化模型保存器类型，固定为"ascendv1_saver"
        - save_directory: 量化模型保存目录，默认为"."
        - part_file_size: 量化模型权重文件大小，默认为4，单位为GB，若part_file_size为0，则不进行分文件保存
        - ext: 扩展配置，用于配置量化模型保存器的扩展功能

    Notes:

    在ascendV1格式中，标准导出件包括：
        - 量化模型描述文件，使用json格式描述量化模型
        - 量化模型权重文件，使用safetensors格式保存量化模型权重（可能包含多个）
        - safetensors index文件，使用json格式保存safetensors文件的索引（可选）

    量化模型描述文件中，包含以下字段：
        - 量化模型权重键值对，键为权重名称，值为权重所属的量化类型
            example:
            {
                "model.layers.0.self_attn.q_proj.weight": "W8A8",
                "model.layers.0.self_attn.q_proj.input_scale": "W8A8",
                "model.layers.0.self_attn.q_proj.input_offset": "W8A8",
                "model.layers.0.self_attn.q_proj.deq_scale": "W8A8",
                "model.layers.0.self_attn.q_proj.quant_bias": "W8A8",
            }

            该例子中表明了model.layers.0.self_attn.q_proj所代表的nn.Linear被量化为W8A8类型，
            该类型由5个参数完全描述。

    量化模型权重文件以safetensors格式保存权重参数。考虑到文件大小的限制，可能会存在多个权重文件，
    此时则会生成safetensors index文件，用于记录各个权重所处的safetensors文件。

    """

    def set_save_directory(self, save_directory: str):
        self.save_directory = str(save_directory)

    type: Literal['ascendv1_saver'] = "ascendv1_saver"
    save_directory: str = Field(default=".", exclude=True)
    part_file_size: int = 4
    ext: Dict[str, Any] = Field(default_factory=dict, exclude_if=lambda v: not v)


ASCENDV1_DESC_JSON_NAME = "quant_model_description.json"
ASCENDV1_SAFETENSORS_NAME = "quant_model_weights.safetensors"


@QABCRegistry.register(dispatch_key=AscendV1Config, abc_class=AutoSessionProcessor)
@logger_setter(prefix='msmodelslim.saver.ascend_v1')  # 4-level: msmodelslim.core.quant_service.modelslim_v1
class AscendV1Saver(AutoSaverProcessor):
    """
    ascendV1 量化模型保存器。该保存器将量化模型保存为AscendV1格式。
    
    关于该格式的更多信息，请参考 AscendV1Config 中的说明。
    """
    # W4A8_DYNAMIC is hidden.
    QUANT_TYPE_PRIORITY = [
        'FLOAT', 'W16A16S', 'W8A8', 'W8A8_DYNAMIC', 'W8A8_MIX', 
        'W4A4_DYNAMIC', 'WFP8AFP8_DYNAMIC', 'W8A8_MXFP8', 'W4A8_MXFP', 'W4A4_MXFP4'
    ]

    def __init__(self, model: nn.Module, config: AscendV1Config, adapter: object, **kwargs: Dict[str, Any]):
        super().__init__(model, config, adapter, **kwargs)
        self.json_append = dict()
        self.metadata = dict()
        self.save_directory = self.get_rank_save_directory() if dist.is_initialized() else config.save_directory
        self.json_writer = JsonWriter(self.save_directory, ASCENDV1_DESC_JSON_NAME)
        self.safetensors_writer = self.get_safetensors_writer(config)
        self.dist_helper: Optional[DistHelper] = None
        self.shared_modules_slice: Optional[List[str]] = None
        self.quarot_info: Optional[qir.QuarotOnlineRotationInfo] = None

        self.version = "1.0.0"
        self.model_quant_type = "Unknown"
        self.group_size = 0

    def support_distributed(self) -> bool:
        return True

    def post_run(self) -> None:

        for name, sub_module in self.model.named_modules(memo=self.processed_modules):
            self.on_float_module(name, sub_module)

        for key, val in self.json_append.items():
            self.json_writer.write(key, val)

        if self.quarot_info is not None:
            self.metadata['quarot'] = self.quarot_info.get_quarot_save_info()

        self.json_writer.write("version", self.version)
        self.json_writer.write("model_quant_type", self.model_quant_type)
        self.json_writer.write("metadata", self.metadata)
        self.json_writer.write("group_size", self.group_size)

        self.json_writer.close()
        self.safetensors_writer.close()

        if not isinstance(self.adapter, IModel):
            raise ToDoError(f'Model Adapter does NOT has attr model_path',
                            action=f'Please implement IModel for saving')
        copy_files(self.adapter.model_path, self.config.save_directory)
        remove_quantization_config(self.config.save_directory)

        if isinstance(self.adapter, AscendV1SaveInterface):
            self.adapter.ascendv1_save_postprocess(self.model, self.config.save_directory)

    def get_safetensors_writer(self, config: AscendV1Config) -> SafetensorsWriter:
        if config.part_file_size > 0:
            return BufferedSafetensorsWriter(
                logger=logger,
                max_gb_size=config.part_file_size,
                save_directory=self.save_directory,
                save_prefix=ASCENDV1_SAFETENSORS_NAME.removesuffix('.safetensors')
            )
        else:
            return SafetensorsWriter(
                logger=logger,
                file_path=os.path.join(self.save_directory, ASCENDV1_SAFETENSORS_NAME),
            )

    def get_rank_save_directory(self) -> str:
        return os.path.join(self.config.save_directory, f"rank_{dist.get_rank()}")

    def write_tensor(self, prefix: str, desc: str, tensor: torch.Tensor):
        self.json_writer.write(prefix, desc)
        self.safetensors_writer.write(prefix, tensor)

    def on_w8a8_static(self, prefix: str, module: qir.W8A8StaticFakeQuantLinear):
        self.update_quant_type("W8A8")

        with torch.device(module.weight.device):
            input_scale, input_offset = module.input_scale, module.input_offset
            input_scale = input_scale.unsqueeze(0) if input_scale.ndim == 0 else input_scale
            input_offset = input_offset.unsqueeze(0) if input_offset.ndim == 0 else input_offset
            weight_scale = module.weight_scale
            quant_weight = module.weight
            deq_scale = input_scale * weight_scale
            deq_scale = deq_scale.squeeze(1) if deq_scale.ndim > 1 else deq_scale
            fp_weight_bias = module.bias if module.bias is not None else torch.zeros(module.weight.shape[0])
            fp_weight_bias = fp_weight_bias.unsqueeze(1) if deq_scale.ndim > 1 else fp_weight_bias
            correction = quant_weight.to(torch.float32).sum(dim=1) * input_offset.to(torch.float32)
            correction = correction.unsqueeze(1) if deq_scale.ndim > 1 else correction
            quant_bias = torch.round(fp_weight_bias / deq_scale - correction).to(torch.int32)
            self.write_tensor(prefix + ".weight", "W8A8", quant_weight.to(torch.int8))
            self.write_tensor(prefix + ".quant_bias", "W8A8", quant_bias.to(torch.int32))
            self.write_tensor(prefix + ".input_scale", "W8A8", input_scale.to(torch.float32))
            self.write_tensor(prefix + ".input_offset", "W8A8", input_offset.to(torch.float32))
            self.write_tensor(prefix + ".deq_scale", "W8A8", deq_scale.to(torch.float32))
            if module.bias is not None:
                self.write_tensor(prefix + ".bias", "W8A8", module.bias.to(torch.float32))

    def on_w8a8_dynamic_per_channel(self, prefix: str, module: qir.W8A8DynamicPerChannelFakeQuantLinear):
        self.update_quant_type("W8A8_DYNAMIC")

        with torch.device(module.weight.device):
            weight_scale = module.weight_scale.unsqueeze(-1)
            self.write_tensor(prefix + ".weight", "W8A8_DYNAMIC", module.weight.to(torch.int8))
            self.write_tensor(prefix + ".weight_scale", "W8A8_DYNAMIC", weight_scale.to(torch.float32))
            self.write_tensor(prefix + ".weight_offset", "W8A8_DYNAMIC",
                              torch.zeros_like(weight_scale).to(torch.float32))
            if module.bias is not None:
                self.write_tensor(prefix + ".bias", "W8A8_DYNAMIC", module.bias.to(torch.float32))

    def on_w8a8_pd_mix(self, prefix: str, module: qir.W8A8PDMixFakeQuantLinear):
        self.update_quant_type("W8A8_MIX")

        with torch.device(module.weight.device):
            input_scale, input_offset = module.input_scale, module.input_offset
            input_scale = input_scale.unsqueeze(0) if input_scale.ndim == 0 else input_scale
            input_offset = input_offset.unsqueeze(0) if input_offset.ndim == 0 else input_offset
            weight_scale = module.weight_scale
            quant_weight = module.weight
            deq_scale = input_scale * weight_scale
            deq_scale = deq_scale.squeeze(1) if deq_scale.ndim > 1 else deq_scale
            fp_weight_bias = module.bias if module.bias is not None else torch.zeros(module.weight.shape[0])
            fp_weight_bias = fp_weight_bias.unsqueeze(1) if deq_scale.ndim > 1 else fp_weight_bias
            correction = quant_weight.to(torch.float32).sum(dim=1) * input_offset.to(torch.float32)
            correction = correction.unsqueeze(1) if deq_scale.ndim > 1 else correction
            quant_bias = torch.round(fp_weight_bias / deq_scale - correction).to(torch.int32)
            self.write_tensor(prefix + ".weight", "W8A8_MIX", quant_weight.to(torch.int8))
            self.write_tensor(prefix + ".quant_bias", "W8A8_MIX", quant_bias.to(torch.int32))
            self.write_tensor(prefix + ".input_scale", "W8A8_MIX", input_scale.to(torch.float32))
            self.write_tensor(prefix + ".input_offset", "W8A8_MIX", input_offset.to(torch.float32))
            self.write_tensor(prefix + ".deq_scale", "W8A8_MIX", deq_scale.to(torch.float32))

            weight_scale = weight_scale.unsqueeze(-1)
            self.write_tensor(prefix + ".weight_scale", "W8A8_MIX", weight_scale.to(torch.float32))
            self.write_tensor(prefix + ".weight_offset", "W8A8_MIX",
                              torch.zeros_like(weight_scale).to(torch.float32))
            if module.bias is not None:
                self.write_tensor(prefix + ".bias", "W8A8_MIX", module.bias.to(torch.float32))

    def on_w8a8_dynamic_per_group(self, prefix: str, module: qir.W8A8DynamicPerGroupFakeQuantLinear):
        self.update_quant_type("W8A8_DYNAMIC")
        self.group_size = module.group_size

        with torch.device(module.weight.device):
            weight_scale = module.weight_scale
            self.write_tensor(prefix + ".weight", "W8A8_DYNAMIC", module.weight.to(torch.int8))
            self.write_tensor(prefix + ".weight_scale", "W8A8_DYNAMIC", weight_scale.to(torch.float32))
            self.write_tensor(prefix + ".weight_offset", "W8A8_DYNAMIC",
                              torch.zeros_like(weight_scale).to(torch.float32))
            if module.bias is not None:
                self.write_tensor(prefix + ".bias", "W8A8_DYNAMIC", module.bias.to(torch.float32))

    def on_wfp8afp8_dynamic_per_channel(self, prefix: str, module: qir.WFP8AFP8DynamicPerChannelFakeQuantLinear):
        self.update_quant_type("WFP8AFP8_DYNAMIC")
        with torch.device(module.weight.device):
            weight_scale = module.weight_scale.unsqueeze(-1)
            self.write_tensor(prefix + ".weight", "WFP8AFP8_DYNAMIC", module.weight.cpu().to(torch.float8_e4m3fn))
            self.write_tensor(prefix + ".weight_scale", "WFP8AFP8_DYNAMIC", weight_scale.to(torch.float32))
            self.write_tensor(prefix + ".weight_offset", "WFP8AFP8_DYNAMIC",
                              torch.zeros_like(weight_scale).to(torch.float32))
            if module.bias is not None:
                self.write_tensor(prefix + ".bias", "WFP8AFP8_DYNAMIC", module.bias.to(torch.float32))

    def on_w8a8_mx_dynamic_per_block(self, prefix: str, module: qir.W8A8MXDynamicPerBlockFakeQuantLinear):
        self.update_quant_type("W8A8_MXFP8")

        with torch.device(module.weight.device):
            if not (isinstance(module.w_axes, (int, list))):
                raise SchemaValidateError("w_axes must be int or list[int].")
            weight_scale = module.weight_scale
            self.group_size = 32
            self.write_tensor(prefix + ".weight", "W8A8_MXFP8", module.weight.cpu().to(torch.float8_e4m3fn))
            self.write_tensor(
                prefix + ".weight_scale",
                "W8A8_MXFP8",
                (weight_scale.squeeze(dim=module.w_axes) + 127).to(torch.uint8)
                # +127 是对 weight_scale 进行偏移处理，使其从-127~128偏移到0~255，正好覆盖torch_uint8的取值范围
            )

    def on_w4a8_dynamic(self, prefix: str, module: qir.W4A8DynamicFakeQuantLinear):
        self.update_quant_type("W4A8_DYNAMIC")

        with torch.device(module.weight.device):
            weight_scale = module.weight_scale.unsqueeze(-1)
            weight = module.weight.to(torch.float32)
            deq_weight = weight.T * module.weight_scale
            scale_bias = process_scale(prefix, deq_weight.T, 16)
            self.write_tensor(prefix + ".weight", "W4A8_DYNAMIC", w4a8_pack_int4(module.weight.to(torch.int8)))
            self.write_tensor(prefix + ".weight_scale", "W4A8_DYNAMIC", weight_scale.to(torch.float32))
            self.write_tensor(prefix + ".weight_offset", "W4A8_DYNAMIC",
                              torch.zeros_like(weight_scale).to(torch.float32))
            self.write_tensor(prefix + '.scale_bias', "W4A8_DYNAMIC", scale_bias.to(torch.float32))
            if module.bias is not None:
                self.write_tensor(prefix + ".bias", "W4A8_DYNAMIC", module.bias.to(torch.float32))

    def on_w4a4_dynamic_per_channel(self, prefix: str, module: qir.W4A4DynamicPerChannelFakeQuantLinear):
        self.update_quant_type("W4A4_DYNAMIC")

        with torch.device(module.weight.device):
            weight_scale = module.weight_scale.unsqueeze(-1)
            weight_offset = module.weight_offset.unsqueeze(-1)
            self.write_tensor(prefix + ".weight", "W4A4_DYNAMIC", module.weight.to(torch.int8))
            self.write_tensor(prefix + ".weight_scale", "W4A4_DYNAMIC", weight_scale.to(torch.float32))
            self.write_tensor(prefix + ".weight_offset", "W4A4_DYNAMIC", weight_offset.to(torch.float32))
            if module.bias is not None:
                self.write_tensor(prefix + ".bias", "W4A4_DYNAMIC", module.bias.to(torch.float32))

    def on_w4a4_dynamic_per_group(self, prefix: str, module: qir.W4A4DynamicPerGroupFakeQuantLinear):
        self.update_quant_type("W4A4_DYNAMIC")
        self.group_size = module.group_size

        with torch.device(module.weight.device):
            self.write_tensor(prefix + ".weight", "W4A4_DYNAMIC", module.weight.to(torch.int8))
            self.write_tensor(prefix + ".weight_scale", "W4A4_DYNAMIC", module.weight_scale.to(torch.float32))
            self.write_tensor(prefix + ".weight_offset", "W4A4_DYNAMIC", module.weight_offset.to(torch.float32))
            if module.bias is not None:
                self.write_tensor(prefix + ".bias", "W4A4_DYNAMIC", module.bias.to(torch.float32))

    def on_w4a4_mx_dynamic_per_block(self, prefix: str, module: qir.W4A4MXDynamicPerBlockFakeQuantLinear):
        self.update_quant_type("W4A4_MXFP4")

        with torch.device(module.weight.device):
            if not (isinstance(module.w_axes, (int, list))):
                raise SchemaValidateError("w_axes must be int or list[int].")
            weight_scale = module.weight_scale
            self.group_size = 32
            self.write_tensor(prefix + ".weight", "W4A4_MXFP4", module.weight.cpu().to(torch.float8_e4m3fn))
            self.write_tensor(
                prefix + ".weight_scale",
                "W4A4_MXFP4",
                (weight_scale.squeeze(dim=module.w_axes) + 127).to(torch.uint8)
                # +127 是对 weight_scale 进行偏移处理，使其从-127~128偏移到0~255，正好覆盖torch_uint8的取值范围
            )

    def on_w4a8_mx_dynamic_per_block(self, prefix: str, module: qir.W4A8MXDynamicPerBlockFakeQuantLinear):
        self.update_quant_type("W4A8_MXFP")

        with torch.device(module.weight.device):
            if not (isinstance(module.w_axes, (int, list))):
                raise SchemaValidateError("w_axes must be int or list[int].")
            weight_scale = module.weight_scale
            self.group_size = 32
            self.write_tensor(prefix + ".weight", "W4A8_MXFP", module.weight.cpu().to(torch.float8_e4m3fn))
            self.write_tensor(
                prefix + ".weight_scale",
                "W4A8_MXFP",
                (weight_scale.squeeze(dim=module.w_axes) + 127).to(torch.uint8)
                # +127 是对 weight_scale 进行偏移处理，使其从-127~128偏移到0~255，正好覆盖torch_uint8的取值范围
            )

    def on_float_linear(self, prefix: str, module: nn.Linear):
        self.update_quant_type("FLOAT")

        return self.on_float_module(prefix, module)

    def on_float_module(self, prefix: str, module: nn.Module):
        for name, param in module.named_parameters(recurse=False, prefix=prefix):
            self.write_tensor(name, "FLOAT", param)

    def on_dynamic_cache(self, prefix: str, module: qir.FakeQuantDynamicCache):
        prefix_list = prefix.split(".")
        prefix_no_last = '.'.join(prefix_list[:-1])
        if "key_states" in prefix_list[-1]:
            self.write_tensor(prefix_no_last + ".k_proj.kv_cache_scale", "C8", module.kv_cache_scale)
            self.write_tensor(prefix_no_last + ".k_proj.kv_cache_offset", "C8", module.kv_cache_offset)
        elif "value_states" in prefix_list[-1]:
            self.write_tensor(prefix_no_last + ".v_proj.kv_cache_scale", "C8", module.kv_cache_scale)
            self.write_tensor(prefix_no_last + ".v_proj.kv_cache_offset", "C8", module.kv_cache_offset)
        else:
            raise ValueError(f"Unknown dynamic cache prefix: {prefix}")
        self.json_append['kv_cache_type'] = "C8"
        self.json_append['kv_quant_type'] = "C8"

    def on_w16a16s(self, prefix: str, module: qir.W16A16sLinear):
        self.update_quant_type("W16A16S")

        for name, param in module.named_parameters(recurse=False, prefix=prefix):
            self.write_tensor(name, "W16A16S", param)

    def on_activation_per_head(self, prefix: str, module: qir.FakeQuantActivationPerHead):
        scale = module.input_scale.to(torch.float32).unsqueeze(-1)
        # 对于1维张量（fa_k.scale, fa_v.scale），转化为2维（与fa_q.scale对齐维数）
        if scale.dim() == 1:
            scale = scale.unsqueeze(-1)
        offset = torch.zeros_like(scale, dtype=torch.int8)
        self.write_tensor(prefix + ".scale", "FAQuant", scale)
        self.write_tensor(prefix + ".offset", "FAQuant", offset)
        self.json_append['fa_quant_type'] = "FAKQuant"

    def on_activation_per_token(self, prefix: str, module: qir.FakeQuantActivationPerToken):
        # 对于FP8 per-token动态量化，保存quant_type标签
        from msmodelslim.ir.qal import QDType, QScope
        if (module.x_q_scheme.dtype == QDType.FP8_E4M3 and 
            module.x_q_scheme.scope == QScope.PER_TOKEN):
            # 保存格式为 self_attn.quant_type，而不是 fa3_q.quant_type
            # 提取父路径，例如 model.layers.0.self_attn.fa3_q -> model.layers.0.self_attn
            parent_prefix = prefix.rsplit('.', 1)[0]
            quant_type_key = parent_prefix + ".quant_type"

            self.json_writer.write(quant_type_key, "FP8_DYNAMIC")
        else:
            raise SchemaValidateError(f"FakeQuantActivationPerToken Unsupported dtype: {module.x_q_scheme.dtype}")

    def on_rotation_wrapper(self, prefix: str, module: qir.QuarotOnlineHeadRotationWrapper):
        """
        处理RotationWrapper类型的模块。

        保存旋转矩阵到model.rotation，并在JSON中添加相应的描述。

        Args:
            prefix: 模块名称前缀
            module: RotationWrapper模块实例
        """
        self.quarot_info = module.rotation_info
        self.safetensors_writer.write(f"{prefix}.heads_rotation", self.quarot_info.heads_rotation.clone())

    def on_kronecker_rotation_wrapper(self, prefix: str, module: qir.QuarotOnlineKroneckerRotationWrapper):
        """
        处理KroneckerRotationWrapper类型的模块。

        保存旋转矩阵到model.rotation_m和model.rotation_n，并在JSON中添加相应的描述。

        Args:
            prefix: 模块名称前缀
            module: KroneckerRotationWrapper模块实例
        """
        self.quarot_info = module.rotation_info
        self.safetensors_writer.write(f"{prefix}.kronecker_rotation_m", self.quarot_info.kronecker_rotation_m.clone())
        self.safetensors_writer.write(f"{prefix}.kronecker_rotation_n", self.quarot_info.kronecker_rotation_n.clone())

    def on_online_rotation_wrapper(self, prefix: str, module: qir.OnlineRotationWrapper):
        """
        处理OnlineRotationWrapper类型的模块。
        """
        rotation_matrix = module.rotation_info.rotation_matrix
        # 保存旋转矩阵，标签为 FLOAT
        self.write_tensor(f"{prefix}", "FLOAT", rotation_matrix.clone())

    def update_quant_type(self, quant_type: str):
        if quant_type not in self.QUANT_TYPE_PRIORITY:
            return
        if self.model_quant_type not in self.QUANT_TYPE_PRIORITY:
            self.model_quant_type = quant_type
            return
        if self.QUANT_TYPE_PRIORITY.index(quant_type) > self.QUANT_TYPE_PRIORITY.index(self.model_quant_type):
            self.model_quant_type = quant_type

    def _process_module(self, prefix: str, module: nn.Module):
        if isinstance(self.adapter, AscendV1SaveInterface):
            self.processed_modules.add(module)
            module = self.adapter.ascendv1_save_module_preprocess(prefix, module, self.model) or module
        super()._process_module(prefix=prefix, module=module)
