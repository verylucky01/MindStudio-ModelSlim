#!/usr/bin/env python
# -*- coding: UTF-8 -*-

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
import os.path
from collections import defaultdict
from functools import lru_cache
from typing import List, Any, Generator, Optional, Tuple, Dict, Union
from unittest.mock import patch

import torch
from safetensors import safe_open
from torch import distributed as dist
from torch import nn
from tqdm import tqdm

from msmodelslim.app.naive_quantization.model_info_interface import ModelInfoInterface
from msmodelslim.core.base.protocol import ProcessRequest
from msmodelslim.core.const import DeviceType
from msmodelslim.core.graph import AdapterConfig, MappingConfig, FusionConfig
from msmodelslim.model.deepseek_v3.quarot import get_ln_fuse_map, get_rotate_map
from msmodelslim.processor.quarot import QuaRotInterface
from msmodelslim.utils.exception import InvalidModelError
from msmodelslim.utils.logging import logger_setter, get_logger
from msmodelslim.utils.security import get_valid_read_path, json_safe_load, MAX_READ_FILE_SIZE_32G
from .convert_fp8_to_bf16 import auto_dequant_state_dict
from .model import Transformer, ModelArgs
from .mtp_quant_module import get_mtp_layer, wrap_mtp_decoder, remove_zero_and_shift
from ..common.layer_wise_forward import generated_decoder_layer_visit_func, TransformersForwardBreak
from ..common.transformers import TransformersModel
from ..interface_hub import ModelSlimPipelineInterfaceV1, FlexSmoothQuantInterface, IterSmoothInterface, AscendV1SaveInterface


@logger_setter("msmodelslim.model.deepseek_v4")
class DeepSeekV4ModelAdapter(TransformersModel,
                              ModelInfoInterface,
                              ModelSlimPipelineInterfaceV1,
                              IterSmoothInterface,
                              FlexSmoothQuantInterface,
                              QuaRotInterface,
                              AscendV1SaveInterface
                              ):
    def get_model_pedigree(self) -> str:
        return 'deepseek_v4'

    def get_model_type(self) -> str:
        return self.model_type

    def handle_dataset(self, dataset: Any, device: DeviceType = DeviceType.NPU) -> List[Any]:
        return self._get_tokenized_data(dataset, device)

    def init_model(self, device: DeviceType = DeviceType.NPU) -> nn.Module:
        torch.set_default_dtype(torch.bfloat16)
        # 如果存在MTP，层数需要+1
        self.config.num_hidden_layers = self.config.num_hidden_layers + 1
        get_logger().info(f"Model with {self.config.num_hidden_layers} layers totally")

        origin = self.config.num_hidden_layers

        self.config.num_hidden_layers = 1
        with torch.device("cpu"):
            model: nn.Module = Transformer(self.config)

        self.config.num_hidden_layers = origin

        state_dict = self.get_state_dict(model)
        auto_dequant_state_dict("", state_dict, str(self.model_path), mtp_layer_prefix=f"layers.{self.config.num_hidden_layers - 1}.")
        model.load_state_dict(state_dict)
        model.eval()
        get_logger().info(f"Create model with {self.config.num_hidden_layers} layers successfully at first")
        return model

    def generate_model_visit(self, model: nn.Module) -> Generator[ProcessRequest, Any, None]:
        return generated_decoder_layer_visit_func(model, transformer_blocks=self.generate_decoder_layer(model))

    def generate_model_forward(self, model: nn.Module, inputs: Any) -> Generator[ProcessRequest, Any, None]:
        # 存储第一个transformer block的输入
        first_block_input: Optional[Tuple] = None

        def break_hook(module: nn.Module, hook_args: Tuple[Any, ...], hook_kwargs: Dict[str, Any]):
            nonlocal first_block_input
            first_block_input = (hook_args, hook_kwargs,)
            raise TransformersForwardBreak()

        remove_handler = model.layers[0].register_forward_pre_hook(break_hook, with_kwargs=True, prepend=True)

        # 执行一次前向传播以获取输入
        try:
            if isinstance(inputs, list) or isinstance(inputs, tuple):
                model(inputs[0])
            elif isinstance(inputs, dict):
                model(**inputs)
            else:
                model(inputs)
        except TransformersForwardBreak:
            pass
        except Exception as e:
            raise e
        finally:
            remove_handler.remove()

        if first_block_input is None:
            raise InvalidModelError("Can't get first block input.", action="Please check the model and input")

        # 循环处理每个transformer block
        current_inputs = first_block_input

        if dist.is_initialized():
            dist.barrier()

        args, kwargs = current_inputs
        h, start_pos, input_ids = args
        for name, block in self.generate_decoder_layer(model):
            if name == f'layers.{self.config.num_hidden_layers - 1}':
                args, kwargs = self.mtp_preprocess(model, mtp_decoder=block, args=args, kwargs=kwargs)
            h = yield ProcessRequest(name, block, args, kwargs)
            args = (h, start_pos, input_ids)

    def mtp_preprocess(self,
                       model: nn.Module,
                       mtp_decoder: nn.Module,
                       args: Tuple[Any, Any],
                       kwargs: Dict[str, Any]) -> Tuple[Tuple[Any, Any], Dict[str, Any]]:
        def wrap_device(module: nn.Module):
            def auto_module(arg):
                module.to('npu')
                result = module(arg.to('npu'))
                module.to('cpu')
                return result

            return auto_module

        pre_hidden_states, start_pos, input_ids = args
        pre_hidden_states = model.hc_head(pre_hidden_states, model.hc_head_fn, model.hc_head_scale, model.hc_head_base)
        # hidden_states = pre_hidden_states = wrap_device(model.norm)(pre_hidden_states)
        hidden_states = wrap_device(model.norm)(pre_hidden_states)
        logits = wrap_device(model.head)(hidden_states[:, -1])
        logits = logits.float()

        ####################### MTP LAYER ######################
        # input_ids = inputs['input_ids'] if isinstance(inputs, dict) else inputs[0]
        input_ids_mtp = remove_zero_and_shift(input_ids)
        position_ids = torch.arange(
            0,
            input_ids_mtp.shape[-1],
            dtype=torch.long,
            device=input_ids.device,
        ) + 1
        position_ids = position_ids.unsqueeze(0)
        input_ids_mtp[:, -1] = logits.argmax(dim=1)

        input_embeds_mtp = wrap_device(mtp_decoder.emb.tok_emb)(input_ids_mtp)
        input_embeds_mtp = wrap_device(mtp_decoder.enorm)(input_embeds_mtp)
        input_embeds_mtp = wrap_device(mtp_decoder.e_proj)(input_embeds_mtp)
        # input_embeds_mtp = wrap_device(mtp_decoder.enorm)(input_embeds_mtp)

        hidden_states_mtp = wrap_device(mtp_decoder.hnorm)(pre_hidden_states)
        hidden_states_mtp = wrap_device(mtp_decoder.h_proj)(hidden_states_mtp)
        # hidden_states_mtp = wrap_device(mtp_decoder.hnorm)(hidden_states_mtp)

        hidden_states_mtp = torch.add(input_embeds_mtp, hidden_states_mtp)
        # hidden_states_mtp = wrap_device(mtp_decoder.norm)(hidden_states_mtp)
        hc_mult = mtp_decoder.hc_head_base.shape[0]
        hidden_states_mtp = hidden_states_mtp.unsqueeze(2).repeat(1, 1, hc_mult, 1) # [b, s, hc_mult, d]

        return (hidden_states_mtp, start_pos+1, input_ids), kwargs

    def enable_kv_cache(self, model: nn.Module, need_kv_cache: bool) -> None:
        pass

    def get_adapter_config_for_subgraph(self) -> List[AdapterConfig]:
        adapter_config = []
        if hasattr(self.config, 'num_experts'):
            expert_num = self.config.num_experts
        elif hasattr(self.config, 'n_routed_experts') and hasattr(self.config, 'n_shared_experts'):
            expert_num = self.config.n_routed_experts

        for layer_idx in range(self.config.num_hidden_layers):
            #============================ MOE =========================
            # MOE FFN 层： Shared Experts
            expert_up_proj = 'layers.' + str(layer_idx) + '.ffn.shared_experts.w3'
            expert_down_proj = 'layers.' + str(layer_idx) + '.ffn.shared_experts.w2'
            up_down_mapping_config_shared = MappingConfig(
                source=expert_up_proj,
                targets=[expert_down_proj]
            )
            adapter_config.extend([
                AdapterConfig(
                    subgraph_type="up-down",
                    mapping=up_down_mapping_config_shared
                )
            ])

            # MOE FFN 层：Routed Experts
            for expert in range(expert_num):
                up_proj = 'layers.' + str(layer_idx) + '.ffn.experts.' + str(expert) + '.w3'
                down_proj = 'layers.' + str(layer_idx) + '.ffn.experts.' + str(expert) + '.w2'
                up_down_mapping_config_expert = MappingConfig(
                    source=up_proj,
                    targets=[down_proj]
                )
                adapter_config.extend([
                    AdapterConfig(
                        subgraph_type="up-down",
                        mapping=up_down_mapping_config_expert
                    )
                ])

            # ======================== Attention ========================
            # Linear-Linear映射配置
            linear1 = f"layers.{layer_idx}.attn.wo_a"
            linear2 = f"layers.{layer_idx}.attn.wo_b"
            wo_mapping_config = MappingConfig(
                source=linear1,
                targets=[linear2]
            )
            adapter_config.extend([
                AdapterConfig(
                    subgraph_type="linear-linear",
                    mapping=wo_mapping_config
                )
            ])

            # 根据层类型添加不同的Attention配置
            if layer_idx < 2 or (layer_idx == self.config.num_hidden_layers - 1):
                # Norm-Linear的映射配置1
                input_norm_mapping_config = MappingConfig(
                    source=f"layers.{layer_idx}.attn_norm",
                    targets=[f"layers.{layer_idx}.attn.wq_a",
                            f"layers.{layer_idx}.attn.wkv",]
                )

                # Norm-Linear的映射配置2：
                qa_norm_mapping_config = MappingConfig(
                    source=f"layers.{layer_idx}.attn.q_norm",
                    targets=[f"layers.{layer_idx}.attn.wq_b",]
                )
            elif layer_idx % 2 == 0:
                # Norm-Linear的映射配置1
                input_norm_mapping_config = MappingConfig(
                    source=f"layers.{layer_idx}.attn_norm",
                    targets=[f"layers.{layer_idx}.attn.wq_a",
                            f"layers.{layer_idx}.attn.wkv",
                            f"layers.{layer_idx}.attn.compressor.wgate", # 2,3,...
                            f"layers.{layer_idx}.attn.compressor.wkv", # 2,3,...
                            f"layers.{layer_idx}.attn.indexer.weights_proj", # 2,4,...
                            f"layers.{layer_idx}.attn.indexer.compressor.wgate", # 2,4,...
                            f"layers.{layer_idx}.attn.indexer.compressor.wkv"] # 2,4,...
                )

                # Norm-Linear的映射配置2：
                qa_norm_mapping_config = MappingConfig(
                    source=f"layers.{layer_idx}.attn.q_norm",
                    targets=[f"layers.{layer_idx}.attn.wq_b",
                            f"layers.{layer_idx}.attn.indexer.wq_b"] # 2,4,...
                )
            else:
                # Norm-Linear的映射配置1
                input_norm_mapping_config = MappingConfig(
                    source=f"layers.{layer_idx}.attn_norm",
                    targets=[f"layers.{layer_idx}.attn.wq_a",
                            f"layers.{layer_idx}.attn.wkv",
                            f"layers.{layer_idx}.attn.compressor.wgate", # 2,3,...
                            f"layers.{layer_idx}.attn.compressor.wkv"] # 2,3,...
                )

                # Norm-Linear的映射配置2：
                qa_norm_mapping_config = MappingConfig(
                    source=f"layers.{layer_idx}.attn.q_norm",
                    targets=[f"layers.{layer_idx}.attn.wq_b",]
                )
                
            adapter_config.extend([
                AdapterConfig(
                    subgraph_type="norm-linear",
                    mapping=input_norm_mapping_config
                ),
                AdapterConfig(
                    subgraph_type="norm-linear",
                    mapping=qa_norm_mapping_config
                ),
            ])

        return adapter_config

    @lru_cache(maxsize=1)
    def get_weight_map(self):
        model_index_path = os.path.join(self.model_path, "model.safetensors.index.json")
        model_index = json_safe_load(model_index_path)
        weight_map = model_index['weight_map']
        return weight_map

    def get_state_dict(self, module: nn.Module, prefix: str = ""):
        weight_map = self.get_weight_map()
        names = map(lambda x: x[0], module.named_parameters())

        groups = defaultdict(list)
        for name in names:
            file_name = weight_map[f'{prefix}.{name}' if prefix else name]
            groups[file_name].append(name)

        state_dict = {}
        for file_name in tqdm(groups, desc=f'Loading {prefix}'):
            file_path = os.path.join(self.model_path, file_name)
            file_path = get_valid_read_path(file_path, extensions='safetensors', size_max=MAX_READ_FILE_SIZE_32G)
            with safe_open(file_path, framework='pt', device='cpu') as f:
                for name in tqdm(groups[file_name], desc=f'Loading {file_path}'):
                    state_dict[name] = f.get_tensor(f'{prefix}.{name}' if prefix else name)
        return state_dict

    def load_mtp_if_not_load(self, mtp_decoder: nn.Module, layer_prefix: str):
        try:
            mtp_decoder.get_submodule('enorm')
        except AttributeError:
            get_logger().info('Creating MTP layer')
            mtp_layer = get_mtp_layer(config=self.config, model_path=self.model_path, layer_prefix=layer_prefix, mtp_layer_prefix=f"layers.{self.config.num_hidden_layers - 1}.")
            wrap_mtp_decoder(mtp_decoder=mtp_decoder, mtp_layer=mtp_layer)
            get_logger().info('Create MTP successfully')

    def load_decoder_if_not_exist(self, model: nn.Module, layer_prefix: str, idx: int):
        try:
            decoder = model.get_submodule(layer_prefix)
        except AttributeError:
            # disable reset_parameters so that the weights will not be initialized
            # these initializations is not necessary because we will load it from the state_dict
            # and these initializations will cost too much time because the DeepSeekV3's decoder layer is too large
            with patch.object(nn.Linear, 'reset_parameters', lambda _self: None):
                get_logger().info(f'Creating decoder layer {idx}')
                module_list: nn.ModuleList = model.layers
                template_module = module_list[0]
                decoder = template_module.__class__(layer_id=idx, args=self.config)

                prefix = layer_prefix if layer_prefix != f"layers.{self.config.num_hidden_layers - 1}" else "mtp.0"
                state_dict = self.get_state_dict(decoder, prefix=prefix)
                auto_dequant_state_dict(layer_prefix, state_dict, str(self.model_path), mtp_layer_prefix=f"layers.{self.config.num_hidden_layers - 1}.")
                decoder.load_state_dict(state_dict)
                decoder.eval()
                module_list.append(decoder)
                get_logger().info(f'Create decoder layer {idx} successfully')
        return decoder

    def generate_decoder_layer(self, model: nn.Module):
        for idx in range(self.config.num_hidden_layers):
            layer_prefix = f"layers.{idx}"
            decoder = self.load_decoder_if_not_exist(model, layer_prefix=layer_prefix, idx=idx)
            if idx == self.config.num_hidden_layers - 1:
                self.load_mtp_if_not_load(decoder, layer_prefix=layer_prefix)
            yield layer_prefix, decoder

    def get_ln_fuse_map(self):
        if hasattr(self.config, 'num_experts'):
            expert_num = self.config.num_experts
        elif hasattr(self.config, 'n_routed_experts') and hasattr(self.config, 'n_shared_experts'):
            expert_num = self.config.n_routed_experts

        pre_ln_linear_map = {}
        #=========================== GLOBAL ==============================
        pre_ln_linear_map['norm'] = ['head']

        ln_linear_map = {}
        for layer_idx in range(self.config.num_hidden_layers):
            #============================ MOE =========================
            ln_linear_map[f"layers.{layer_idx}.ffn_norm"] = [
                f"layers.{layer_idx}.ffn.experts.{i}.{proj}"
                for proj in ["w1", "w3"]
                for i in range(expert_num)
            ]
            # shared experts
            ln_linear_map[f"layers.{layer_idx}.ffn_norm"] += [
                f"layers.{layer_idx}.ffn.shared_experts.{proj}"
                for proj in ["w1", "w3"]
            ]
            # expert gate
            ln_linear_map[f"layers.{layer_idx}.ffn_norm"] += [
                f"layers.{layer_idx}.ffn.gate"
            ]

            #============================ Attention =========================
            # 根据层类型添加不同的Attention配置
            if layer_idx < 2 or (layer_idx == self.config.num_hidden_layers - 1):
                # Norm-Linear的映射配置1
                ln_linear_map[f"layers.{layer_idx}.attn_norm"] = [
                    f"layers.{layer_idx}.attn.wq_a",
                    f"layers.{layer_idx}.attn.wkv",
                ]
                # Norm-Linear的映射配置2：
                ln_linear_map[f"layers.{layer_idx}.attn.q_norm"] = [
                    f"layers.{layer_idx}.attn.wq_b",
                ]
            elif layer_idx % 2 == 0:
                # Norm-Linear的映射配置1
                ln_linear_map[f"layers.{layer_idx}.attn_norm"] = [
                    f"layers.{layer_idx}.attn.wq_a",
                    f"layers.{layer_idx}.attn.wkv",
                    f"layers.{layer_idx}.attn.compressor.wgate", # 2,3,...
                    f"layers.{layer_idx}.attn.compressor.wkv", # 2,3,...
                    f"layers.{layer_idx}.attn.indexer.weights_proj", # 2,4,...
                    f"layers.{layer_idx}.attn.indexer.compressor.wgate", # 2,4,...
                    f"layers.{layer_idx}.attn.indexer.compressor.wkv", # 2,4,...
                ]

                # Norm-Linear的映射配置2：
                ln_linear_map[f"layers.{layer_idx}.attn.q_norm"] = [
                    f"layers.{layer_idx}.attn.wq_b",
                    f"layers.{layer_idx}.attn.indexer.wq_b",] # 2,4,...
            else:
                # Norm-Linear的映射配置1
                ln_linear_map[f"layers.{layer_idx}.attn_norm"] = [
                    f"layers.{layer_idx}.attn.wq_a",
                    f"layers.{layer_idx}.attn.wkv",
                    f"layers.{layer_idx}.attn.compressor.wgate", # 2,3,...
                    f"layers.{layer_idx}.attn.compressor.wkv", # 2,3,...
                ]
                
                # Norm-Linear的映射配置2：
                ln_linear_map[f"layers.{layer_idx}.attn.q_norm"] = [
                    f"layers.{layer_idx}.attn.wq_b",
                ]
        #============================ MTP =========================
        layer_idx = self.config.num_hidden_layers - 1
        # MTP 的 norm-linear 融合
        # Norm-Linear的映射配置1 （MTP专有）：
        ln_linear_map[f"layers.{layer_idx}.enorm"] = [
            f"layers.{layer_idx}.e_proj",
        ]
        # Norm-Linear的映射配置2 （MTP专有）：
        ln_linear_map[f"layers.{layer_idx}.hnorm"] = [
            f"layers.{layer_idx}.h_proj",
        ]
        # Norm-Linear的映射配置3 （MTP专有）：
        ln_linear_map[f"layers.{layer_idx}.norm"] = [
            f"layers.{layer_idx}.head",
        ]

        return pre_ln_linear_map, ln_linear_map

    def get_bake_names(self):
        return [], []

    def get_rotate_map(self, block_size):
        if hasattr(self.config, 'num_experts'):
            expert_num = self.config.num_experts
        elif hasattr(self.config, 'n_routed_experts') and hasattr(self.config, 'n_shared_experts'):
            expert_num = self.config.n_routed_experts

        rot_pairs = {}

        # chain rot
        rot = QuaRotInterface.get_rotate_command(
            size=self.config.dim,
            mode=QuaRotInterface.QuaRotMode.HADAMARD,
            block_size=block_size,
        )
        # ============================= GLOBAL ==============================
        pre_left_rot = {}
        pre_right_rot = {}
        pre_right_rot[f'embed'] = rot
        pre_right_rot[f'hc_head_fn'] = rot
        pre_right_rot[f'head'] = rot
        pre_run = QuaRotInterface.RotatePair(left_rot=pre_left_rot, right_rot=pre_right_rot)

        left_rot = {}
        right_rot = {}
        for layer_idx in range(self.config.num_hidden_layers):
            #============================ MOE =========================
            for i in range(expert_num):
                left_rot[f"layers.{layer_idx}.ffn.experts.{i}.w2"] = rot
            left_rot[f"layers.{layer_idx}.ffn.shared_experts.w2"] = rot

            right_rot[f"layers.{layer_idx}.hc_ffn_fn"] = rot
            for i in range(expert_num):
                right_rot[f"layers.{layer_idx}.ffn.experts.{i}.w1"] = rot
                right_rot[f"layers.{layer_idx}.ffn.experts.{i}.w3"] = rot
            right_rot[f"layers.{layer_idx}.ffn.shared_experts.w1"] = rot
            right_rot[f"layers.{layer_idx}.ffn.shared_experts.w3"] = rot
            right_rot[f"layers.{layer_idx}.ffn.gate"] = rot

            #============================ Attention =========================
            left_rot[f"layers.{layer_idx}.attn.wo_b"] = rot
            right_rot[f"layers.{layer_idx}.hc_attn_fn"] = rot
            if layer_idx < 2 or (layer_idx == self.config.num_hidden_layers - 1):
                right_rot[f"layers.{layer_idx}.attn.wq_a"] = rot
                right_rot[f"layers.{layer_idx}.attn.wkv"] = rot
            elif layer_idx % 2 == 0:
                right_rot[f"layers.{layer_idx}.attn.wq_a"] = rot
                right_rot[f"layers.{layer_idx}.attn.wkv"] = rot
                right_rot[f"layers.{layer_idx}.attn.compressor.wgate"] = rot
                right_rot[f"layers.{layer_idx}.attn.compressor.wkv"] = rot
                right_rot[f"layers.{layer_idx}.attn.indexer.weights_proj"] = rot
                right_rot[f"layers.{layer_idx}.attn.indexer.compressor.wgate"] = rot
                right_rot[f"layers.{layer_idx}.attn.indexer.compressor.wkv"] = rot
            else:
                right_rot[f"layers.{layer_idx}.attn.wq_a"] = rot
                right_rot[f"layers.{layer_idx}.attn.wkv"] = rot
                right_rot[f"layers.{layer_idx}.attn.compressor.wgate"] = rot
                right_rot[f"layers.{layer_idx}.attn.compressor.wkv"] = rot
        #============================ MTP =========================
        layer_idx = (self.config.num_hidden_layers - 1)
        right_rot[f"layers.{layer_idx}.h_proj"] = rot
        right_rot[f"layers.{layer_idx}.head"] = rot

        right_rot[f"layers.{layer_idx}.emb.tok_emb"] = rot
        right_rot[f"layers.{layer_idx}.e_proj"] = rot

        left_rot[f"layers.{layer_idx}.e_proj"] = rot
        left_rot[f"layers.{layer_idx}.h_proj"] = rot
        
        rot_pairs['rot'] = QuaRotInterface.RotatePair(left_rot=left_rot, right_rot=right_rot)

        # q_b_proj rot
        rot_b_proj = QuaRotInterface.get_rotate_command(
            size=self.config.q_lora_rank,
            mode=QuaRotInterface.QuaRotMode.BLOCK_HADAMARD_SHIFTED,
            block_size=block_size,
        )
        left_rot_b_proj = {}
        right_rot_b_proj = {}
        for layer_idx in range(self.config.num_hidden_layers):
            # =============================== Attention =========================
            left_rot_b_proj[f"layers.{layer_idx}.attn.wq_a"] = rot_b_proj
            if layer_idx < 2 or (layer_idx == self.config.num_hidden_layers - 1):
                right_rot_b_proj[f"layers.{layer_idx}.attn.wq_b"] = rot_b_proj
            elif layer_idx % 2 == 0:
                right_rot_b_proj[f"layers.{layer_idx}.attn.wq_b"] = rot_b_proj
                right_rot_b_proj[f"layers.{layer_idx}.attn.indexer.wq_b"] = rot_b_proj
            else:
                right_rot_b_proj[f"layers.{layer_idx}.attn.wq_b"] = rot_b_proj
        rot_pairs["rot_b_proj"] = QuaRotInterface.RotatePair(left_rot=left_rot_b_proj, right_rot=right_rot_b_proj)
        return [pre_run], [pair for pair in rot_pairs.values()]

    def ascendv1_save_module_preprocess(self, prefix: str, module: nn.Module, model: nn.Module) -> Tuple[str, nn.Module]:
        mtp_layer_prefix = f"layers.{self.config.num_hidden_layers - 1}"
        if prefix.startswith(mtp_layer_prefix):
            prefix = prefix.replace(mtp_layer_prefix, "mtp.0")

        return prefix, module

    def _load_config(self, trust_remote_code=False) -> object:
        return ModelArgs()
