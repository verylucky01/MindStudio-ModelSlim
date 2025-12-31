#  Copyright c) Huawei Technologies Co. Ltd. 2025-2025
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
from .convert_fp8_to_bf16 import auto_convert_module_fp8_to_bf16
from .model import Transformer, ModelArgs
from .mtp_quant_module import get_mtp_layer, wrap_mtp_decoder, remove_zero_and_shift
from ..common.layer_wise_forward import generated_decoder_layer_visit_func, TransformersForwardBreak
from ..common.transformers import TransformersModel
from ..interface_hub import ModelSlimPipelineInterfaceV1, FlexSmoothQuantInterface


@logger_setter("msmodelslim.model.deepseek_v3_2")
class DeepSeekV32ModelAdapter(TransformersModel,
                              ModelInfoInterface,
                              ModelSlimPipelineInterfaceV1,
                              FlexSmoothQuantInterface,
                              QuaRotInterface
                              ):
    def get_model_pedigree(self) -> str:
        return 'deepseek_v3_2'

    def get_model_type(self) -> str:
        return self.model_type

    def handle_dataset(self, dataset: Any, device: DeviceType = DeviceType.NPU) -> List[Any]:
        return self._get_tokenized_data(dataset, device)

    def init_model(self, device: DeviceType = DeviceType.NPU) -> nn.Module:
        torch.set_default_dtype(torch.bfloat16)
        self.config.num_hidden_layers = 62
        get_logger().info(f"Model with {self.config.num_hidden_layers} layers totally")

        origin = self.config.num_hidden_layers

        self.config.num_hidden_layers = 1
        with torch.device("cpu"):
            model: nn.Module = Transformer(self.config)

        self.config.num_hidden_layers = origin

        state_dict = self.get_state_dict(model)
        model.load_state_dict(state_dict)
        auto_convert_module_fp8_to_bf16("", model, str(self.model_path))
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

        remove_handler = model.model.layers[0].register_forward_pre_hook(break_hook, with_kwargs=True, prepend=True)

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
        for name, block in self.generate_decoder_layer(model):
            if name == f'model.layers.{self.config.num_hidden_layers - 1}':
                args, kwargs = self.mtp_preprocess(model, mtp_decoder=block, inputs=inputs, args=args, kwargs=kwargs)
            h, residual = yield ProcessRequest(name, block, args, kwargs)
            args = (h, residual)

    def mtp_preprocess(self,
                       model: nn.Module,
                       mtp_decoder: nn.Module,
                       inputs: Union[List[Any], Dict[str, Any]],
                       args: Tuple[Any, Any],
                       kwargs: Dict[str, Any]) -> Tuple[Tuple[Any, Any], Dict[str, Any]]:
        def wrap_device(module: nn.Module):
            def auto_module(arg):
                module.to('npu')
                result = module(arg.to('npu'))
                module.to('cpu')
                return result

            return auto_module

        pre_hidden_states, residual = args
        hidden_states = model.model.norm(pre_hidden_states)
        logits = wrap_device(model.lm_head)(hidden_states)
        logits = logits.float()

        ####################### MTP LAYER ######################
        input_ids = inputs['input_ids'] if isinstance(inputs, dict) else inputs[0]
        input_ids_mtp = remove_zero_and_shift(input_ids)
        position_ids = torch.arange(
            0,
            input_ids_mtp.shape[-1],
            dtype=torch.long,
            device=input_ids.device,
        ) + 1
        position_ids = position_ids.unsqueeze(0)
        logits[:, -1, :].argmax(dim=1)
        input_ids_mtp[:, -1] = logits[:, -1, :].argmax(dim=1)

        input_embeds_mtp = wrap_device(mtp_decoder.embed_tokens)(input_ids_mtp)
        input_embeds_mtp = wrap_device(mtp_decoder.enorm)(input_embeds_mtp)
        hidden_states_mtp = wrap_device(mtp_decoder.hnorm)(pre_hidden_states)
        hidden_states_mtp = torch.cat([input_embeds_mtp, hidden_states_mtp], dim=-1)
        hidden_states_mtp = wrap_device(mtp_decoder.eh_proj)(hidden_states_mtp)

        attention_mask = inputs['attention_mask'] if isinstance(inputs, dict) else inputs[1]

        # transformers==4.48.2
        from transformers.modeling_attn_mask_utils import _prepare_4d_causal_attention_mask

        attention_mask_mtp = _prepare_4d_causal_attention_mask(
            attention_mask,
            (input_ids.shape[:2]),
            input_embeds_mtp,
            0,
        )

        start_pos = kwargs['start_pos'] + 1
        seq_len = len(kwargs['freqs_cis'])
        kwargs['mask'] = attention_mask_mtp.squeeze(1)
        kwargs['freqs_cis'] = model.model.freqs_cis[start_pos: start_pos + seq_len]
        return (hidden_states_mtp, residual), kwargs

    def enable_kv_cache(self, model: nn.Module, need_kv_cache: bool) -> None:
        pass

    def get_adapter_config_for_subgraph(self) -> List[AdapterConfig]:
        adapter_config = []
        if hasattr(self.config, 'num_experts'):
            expert_num = self.config.num_experts
        elif hasattr(self.config, 'n_routed_experts') and hasattr(self.config, 'n_shared_experts'):
            expert_num = self.config.n_routed_experts

        for layer_idx in range(self.config.num_hidden_layers):
            # OKV_b融合的映射配置：o_proj -> kv_b_proj
            okv_b_mapping_config = MappingConfig(
                source=f"model.layers.{layer_idx}.self_attn.kv_b_proj",  # KV_b投影层
                targets=[f"model.layers.{layer_idx}.self_attn.o_proj"]  # 输出投影层
            )

            # Norm-Linear融合的映射配置1：q_a_proj, kv_a_proj_with_mqa -> input_layernorm
            input_norm_mapping_config = MappingConfig(
                source=f"model.layers.{layer_idx}.input_layernorm",  # 第一个LayerNorm
                targets=[f"model.layers.{layer_idx}.self_attn.q_a_proj",
                         f"model.layers.{layer_idx}.self_attn.kv_a_proj_with_mqa",
                         f"model.layers.{layer_idx}.self_attn.indexer.wk",
                         f"model.layers.{layer_idx}.self_attn.indexer.weights_proj"]  # 注意力层的Q_a,KV_a投影
            )

            # Norm-Linear融合的映射配置2：q_b_proj -> q_a_layernorm
            qa_norm_mapping_config = MappingConfig(
                source=f"model.layers.{layer_idx}.self_attn.q_a_layernorm",  # q_a_layernorm
                targets=[f"model.layers.{layer_idx}.self_attn.q_b_proj",
                         f"model.layers.{layer_idx}.self_attn.indexer.wq_b"]  # q_b投影
            )

            # 为当前layer添加4个配置
            adapter_config.extend([
                AdapterConfig(
                    subgraph_type="ov",
                    mapping=okv_b_mapping_config,
                    extra_config={
                        'group_method': 'max'
                    },
                    fusion=FusionConfig(
                        fusion_type="kv",
                        num_attention_heads=self.config.num_attention_heads,
                        num_key_value_heads=self.config.num_key_value_heads,
                        custom_config={
                            'qk_nope_head_dim': self.config.qk_nope_head_dim,
                            'v_head_dim': self.config.v_head_dim,
                        }
                    ),
                ),
                AdapterConfig(
                    subgraph_type="norm-linear",
                    mapping=input_norm_mapping_config
                ),
                AdapterConfig(
                    subgraph_type="norm-linear",
                    mapping=qa_norm_mapping_config
                ),
            ])

            # 根据层类型添加不同的FFN配置
            if layer_idx < self.config.first_k_dense_replace:
                # Dense FFN 层
                up_proj = 'model.layers.' + str(layer_idx) + '.mlp.up_proj'
                down_proj = 'model.layers.' + str(layer_idx) + '.mlp.down_proj'
                up_down_mapping_config = MappingConfig(
                    source=up_proj,  # 上投影层
                    targets=[down_proj]  # 下投影层
                )
                adapter_config.extend([
                    AdapterConfig(
                        subgraph_type="up-down",
                        mapping=up_down_mapping_config
                    ),
                ])
            else:
                # MOE FFN 层：Shared Experts
                expert_up_proj = 'model.layers.' + str(layer_idx) + '.mlp.shared_experts.up_proj'
                expert_down_proj = 'model.layers.' + str(layer_idx) + '.mlp.shared_experts.down_proj'
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
                    up_proj = 'model.layers.' + str(layer_idx) + '.mlp.experts.' + str(expert) + '.up_proj'
                    down_proj = 'model.layers.' + str(layer_idx) + '.mlp.experts.' + str(expert) + '.down_proj'
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

    def load_mtp_if_not_load(self, mtp_decoder: nn.Module):
        try:
            mtp_decoder.get_submodule('shared_head')
        except AttributeError:
            get_logger().info('Creating MTP layer')
            mtp_layer = get_mtp_layer(config=self.config, model_path=self.model_path)
            wrap_mtp_decoder(mtp_decoder=mtp_decoder, mtp_layer=mtp_layer)
            get_logger().info('Create MTP successfully')

    def load_decoder_if_not_exist(self, model: nn.Module, name: str, idx: int):
        try:
            decoder = model.get_submodule(name)
        except AttributeError:
            # disable reset_parameters so that the weights will not be initialized
            # these initializations is not necessary because we will load it from the state_dict
            # and these initializations will cost too much time because the DeepSeekV3's decoder layer is too large
            with patch.object(nn.Linear, 'reset_parameters', lambda _self: None):
                get_logger().info(f'Creating decoder layer {idx}')
                module_list: nn.ModuleList = model.model.layers
                template_module = module_list[0]
                decoder = template_module.__class__(layer_id=idx, args=self.config)

                state_dict = self.get_state_dict(decoder, prefix=name)
                decoder.load_state_dict(state_dict)
                auto_convert_module_fp8_to_bf16(name, decoder, str(self.model_path))
                decoder.eval()
                module_list.append(decoder)
                get_logger().info(f'Create decoder layer {idx} successfully')
        return decoder

    def generate_decoder_layer(self, model: nn.Module):
        for idx in range(self.config.num_hidden_layers):
            name = f"model.layers.{idx}"
            decoder = self.load_decoder_if_not_exist(model, name=name, idx=idx)
            if idx == self.config.num_hidden_layers - 1:
                self.load_mtp_if_not_load(decoder)
            yield name, decoder

    def get_ln_fuse_map(self):
        ln_linear_map = get_ln_fuse_map(self.config, num_hidden_layers=self.config.num_hidden_layers)
        for layer_idx in range(self.config.num_hidden_layers):
            ln_linear_map[f"model.layers.{layer_idx}.input_layernorm"].append(
                f"model.layers.{layer_idx}.self_attn.indexer.wk",
            )
            ln_linear_map[f"model.layers.{layer_idx}.input_layernorm"].append(
                f"model.layers.{layer_idx}.self_attn.indexer.weights_proj",
            )
            ln_linear_map[f"model.layers.{layer_idx}.self_attn.q_a_layernorm"].append(
                f"model.layers.{layer_idx}.self_attn.indexer.wq_b"
            )
        return {}, ln_linear_map

    def get_bake_names(self):
        return [], []

    def get_rotate_map(self, block_size):
        pre_run, rot_pairs, rotate_matrix = get_rotate_map(self.config,
                                                           block_size,
                                                           num_hidden_layers=self.config.num_hidden_layers)
        for layer_idx in range(self.config.num_hidden_layers):
            rot_pairs['rot'].right_rot[f"model.layers.{layer_idx}.self_attn.indexer.wk"] = rotate_matrix['rot']
            rot_pairs['rot'].right_rot[f"model.layers.{layer_idx}.self_attn.indexer.weights_proj"] = \
                rotate_matrix['rot']
            rot_pairs['rot_b_proj'].right_rot[f"model.layers.{layer_idx}.self_attn.indexer.wq_b"] = \
                rotate_matrix['rot_b_proj']
        return [pre_run], [pair for pair in rot_pairs.values()]

    def _load_config(self, trust_remote_code=False) -> object:
        return ModelArgs()
