# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.

from typing import List, Any, Generator

from torch import nn

from msmodelslim.core.base.protocol import ProcessRequest
from msmodelslim.core.const import DeviceType
from msmodelslim.core.graph.adapter_types import AdapterConfig, MappingConfig
from msmodelslim.quant.processor.kv_smooth import KVSmoothFusedType, KVSmoothFusedUnit
from msmodelslim.quant.processor.quarot import (
    QuaRotInterface,
    QuaRotOnlineInterface
)
from msmodelslim.utils.exception import InvalidModelError
from msmodelslim.utils.logging import logger_setter, get_logger
from ..common.layer_wise_forward import generated_decoder_layer_visit_func, transformers_generated_forward_func
from ..common.transformers import TransformersModel
from ..interface_hub import ModelInfoInterface, ModelSlimPipelineInterfaceV0, ModelSlimPipelineInterfaceV1, \
    AnalyzePipelineInterface, KVSmoothFusedInterface, SmoothQuantInterface, IterSmoothInterface, \
    FlexSmoothQuantInterface


@logger_setter()
class Qwen3ModelAdapter(TransformersModel,
                        ModelInfoInterface,
                        ModelSlimPipelineInterfaceV0,
                        ModelSlimPipelineInterfaceV1,
                        AnalyzePipelineInterface,
                        KVSmoothFusedInterface,
                        SmoothQuantInterface,
                        IterSmoothInterface,
                        FlexSmoothQuantInterface,
                        QuaRotInterface,
                        QuaRotOnlineInterface
                        ):
    def get_model_type(self) -> str:
        return self.model_type

    def get_model_pedigree(self) -> str:
        return 'qwen3'

    def load_model(self, device: DeviceType = DeviceType.NPU) -> nn.Module:
        return self._load_model(device)

    def handle_dataset(self, dataset: Any, device: DeviceType = DeviceType.NPU) -> List[Any]:
        return self._get_tokenized_data(dataset, device)

    def handle_dataset_by_batch(self,
                                dataset: Any,
                                batch_size: int,
                                device: DeviceType = DeviceType.NPU) -> List[Any]:
        return self._get_batch_tokenized_data(calib_list=dataset,
                                              batch_size=batch_size,
                                              device=device)

    def init_model(self, device: DeviceType = DeviceType.NPU) -> nn.Module:
        return self._load_model(device)

    def generate_model_visit(self, model: nn.Module) -> Generator[ProcessRequest, Any, None]:
        yield from generated_decoder_layer_visit_func(model)

    def generate_model_forward(self, model: nn.Module, inputs: Any,
                               ) -> Generator[ProcessRequest, Any, None]:
        yield from transformers_generated_forward_func(model, inputs)

    def enable_kv_cache(self, model: nn.Module, need_kv_cache: bool) -> None:
        return self._enable_kv_cache(model, need_kv_cache)

    def get_kvcache_smooth_fused_subgraph(self) -> List[KVSmoothFusedUnit]:
        return [
            KVSmoothFusedUnit(
                attention_name=f"model.layers.{i}.self_attn",
                layer_idx=i,
                fused_from_query_states_name="q_norm",
                fused_from_key_states_name="k_norm",
                fused_type=KVSmoothFusedType.StateViaRopeToNorm
            )
            for i in range(self.config.num_hidden_layers)
        ]

    def get_head_dim(self) -> int:
        if hasattr(self.config, 'head_dim'):
            return self.config.head_dim

        get_logger().warning(f'head_dim is not found in config.json, use hidden_size // num_attention_heads instead')
        if not hasattr(self.config, 'hidden_size'):
            raise InvalidModelError("hidden_size is not found in config.json",
                                    action="Please check the model config.json")
        if not hasattr(self.config, 'num_attention_heads'):
            raise InvalidModelError("num_attention_heads is not found in config.json",
                                    action="Please check the model config.json")
        if self.config.num_attention_heads == 0:
            raise InvalidModelError("num_attention_heads is 0 in config.json, which should be greater than 0",
                                    action="Please check the model config.json")
        return self.config.hidden_size // self.config.num_attention_heads

    def get_num_key_value_groups(self) -> int:
        if not hasattr(self.config, 'num_attention_heads'):
            raise InvalidModelError("num_attention_heads is not found in config.json",
                                    action=f"Please check config.json in {self.model_path}")
        if not hasattr(self.config, 'num_key_value_heads'):
            raise InvalidModelError("num_key_value_heads is not found in config.json",
                                    action=f"Please check config.json in {self.model_path}")
        if self.config.num_key_value_heads == 0:
            raise InvalidModelError("num_key_value_heads is 0 in config.json, which should be greater than 0",
                                    action=f"Please check config.json in {self.model_path}")
        return self.config.num_attention_heads // self.config.num_key_value_heads

    def get_num_key_value_heads(self) -> int:
        if not hasattr(self.config, 'num_key_value_heads'):
            raise InvalidModelError("num_key_value_heads is not found in config.json",
                                    action=f"Please check config.json in {self.model_path}")
        return self.config.num_key_value_heads

    def get_adapter_config_for_subgraph(self) -> List[AdapterConfig]:
        adapter_config = []
        for layer_idx in range(self.config.num_hidden_layers):
            # Norm-Linear的映射配置1：输入层归一化到QKV投影
            norm_linear_mapping_config1 = MappingConfig(
                source=f"model.layers.{layer_idx}.input_layernorm",  # 第一个LayerNorm
                targets=[f"model.layers.{layer_idx}.self_attn.k_proj",
                         f"model.layers.{layer_idx}.self_attn.q_proj",
                         f"model.layers.{layer_idx}.self_attn.v_proj"]  # 注意力层的QKV投影
            )

            # Norm-Linear的映射配置2：后注意力层归一化到MLP投影
            norm_linear_mapping_config2 = MappingConfig(
                source=f"model.layers.{layer_idx}.post_attention_layernorm",  # 第二个LayerNorm
                targets=[f"model.layers.{layer_idx}.mlp.gate_proj",
                         f"model.layers.{layer_idx}.mlp.up_proj"]  # MLP层的门控和上投影
            )

            # OV的映射配置（QKV到输出投影）
            ov_mapping_config = MappingConfig(
                source=f"model.layers.{layer_idx}.self_attn.v_proj",  # V投影层
                targets=[f"model.layers.{layer_idx}.self_attn.o_proj"]  # 输出投影层
            )

            # Up-Down的映射配置
            up_down_mapping_config = MappingConfig(
                source=f"model.layers.{layer_idx}.mlp.up_proj",  # 上投影层
                targets=[f"model.layers.{layer_idx}.mlp.down_proj"]  # 下投影层
            )

            # 为当前layer添加4个配置
            adapter_config.extend([
                AdapterConfig(
                    subgraph_type="norm-linear",
                    mapping=norm_linear_mapping_config1
                ),
                AdapterConfig(
                    subgraph_type="norm-linear",
                    mapping=norm_linear_mapping_config2
                ),
                AdapterConfig(
                    subgraph_type="ov",
                    mapping=ov_mapping_config,
                    extra_config={
                        'group_method': 'max'
                    }
                ),
                AdapterConfig(
                    subgraph_type="up-down",
                    mapping=up_down_mapping_config
                )
            ])
        return adapter_config

    def get_hidden_dim(self):
        return self.config.hidden_size

    def get_num_attention_heads(self):
        return self.config.num_attention_heads

    def get_lm_head(self) -> str:
        return "lm_head"

    def get_pre_head_layernorm(self) -> str:
        return "model.norm"

    def get_embedding(self) -> str:
        return "model.embed_tokens"

    def get_layer_wise_norm_liner_pair(self, decoder_module: nn.Module):
        norm_linear_pairs = {decoder_module.input_layernorm: [decoder_module.self_attn.q_proj,
                                                              decoder_module.self_attn.k_proj,
                                                              decoder_module.self_attn.v_proj],
                             decoder_module.post_attention_layernorm: [decoder_module.mlp.gate_proj,
                                                                       decoder_module.mlp.up_proj]}
        return norm_linear_pairs

    def get_layer_wise_ov_pair(self, decoder_module: nn.Module):
        ov_pairs = {decoder_module.self_attn.o_proj: decoder_module.self_attn.v_proj}
        return ov_pairs

    def get_layer_wise_up_down_pair(self, decoder_module: nn.Module):
        up_down_pairs = {decoder_module.mlp.up_proj: decoder_module.mlp.down_proj}
        return up_down_pairs

    def get_ln_fuse_map(self):
        return {}, qwen3_get_ln_fuse_map(self.config)

    def get_bake_names(self):
        return [], []

    def get_rotate_map(self, block_size):
        pre_run, rot_pairs, _, _ = qwen3_get_rotate_map(self.config, block_size)
        return [pre_run], [pair for pair in rot_pairs.values()]


def qwen3_get_ln_fuse_map(config):
    # for quarot rotate interface
    ln_linear_map = {}
    for layer_idx in range(config.num_hidden_layers):
        ln_linear_map[f"model.layers.{layer_idx}.input_layernorm"] = [
            f"model.layers.{layer_idx}.self_attn.q_proj",
            f"model.layers.{layer_idx}.self_attn.k_proj",
            f"model.layers.{layer_idx}.self_attn.v_proj"
        ]

        # mlp
        ln_linear_map[f"model.layers.{layer_idx}.post_attention_layernorm"] = [
            f"model.layers.{layer_idx}.mlp.{proj}"
            for proj in ["gate_proj", "up_proj"]
        ]
    ln_linear_map["model.norm"] = ['lm_head']
    return ln_linear_map


def qwen3_get_rotate_map(config, block_size):
    rot = QuaRotInterface.get_rotate_command(
        size=config.hidden_size,
        block_size=block_size,
        mode=QuaRotInterface.QuaRotMode.HADAMARD,
    )
    rot_uv = QuaRotInterface.get_rotate_command(
        size=config.head_dim,
        block_size=block_size,
        mode=QuaRotInterface.QuaRotMode.HADAMARD,
    )
    # pre run 
    left_rot = {}
    right_rot = {}
    # embedding weight is transposed, right is output channel
    right_rot[f"model.embed_tokens"] = rot
    pre_run = QuaRotInterface.RotatePair(left_rot=left_rot, right_rot=right_rot)
    rot_pairs = {}
    # rot
    left_rot = {}
    right_rot = {}
    right_rot[f"lm_head"] = rot
    for layer_idx in range(config.num_hidden_layers):
        right_rot[f"model.layers.{layer_idx}.self_attn.q_proj"] = rot
        right_rot[f"model.layers.{layer_idx}.self_attn.k_proj"] = rot
        right_rot[f"model.layers.{layer_idx}.self_attn.v_proj"] = rot
        left_rot[f"model.layers.{layer_idx}.self_attn.o_proj"] = rot
        # mlp
        right_rot[f"model.layers.{layer_idx}.mlp.gate_proj"] = rot
        right_rot[f"model.layers.{layer_idx}.mlp.up_proj"] = rot
        left_rot[f"model.layers.{layer_idx}.mlp.down_proj"] = rot
    rot_pairs['rot'] = QuaRotInterface.RotatePair(left_rot=left_rot, right_rot=right_rot)

    # rot_uv
    left_rot_uv = {}
    right_rot_uv = {}
    for layer_idx in range(config.num_hidden_layers):
        left_rot_uv[f"model.layers.{layer_idx}.self_attn.v_proj"] = rot_uv
        right_rot_uv[f"model.layers.{layer_idx}.self_attn.o_proj"] = rot_uv
    rot_pairs["rot_uv"] = QuaRotInterface.RotatePair(left_rot=left_rot_uv, right_rot=right_rot_uv)

    return pre_run, rot_pairs, rot, rot_uv
