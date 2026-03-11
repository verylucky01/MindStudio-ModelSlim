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

import os
from collections import defaultdict
from functools import lru_cache
from pathlib import Path
from typing import List, Any, Generator, Tuple, Dict
from unittest.mock import patch

import torch
from safetensors import safe_open
from torch import nn
from tqdm import tqdm
from transformers import AutoProcessor, Glm4vMoeForConditionalGeneration
from transformers.models.glm4v_moe.modeling_glm4v_moe import Glm4vMoeTextDecoderLayer, Glm4vMoeTextMoE
from transformers.masking_utils import create_causal_mask

from msmodelslim.core.const import DeviceType
from msmodelslim.app.naive_quantization.model_info_interface import ModelInfoInterface
from msmodelslim.core.base.protocol import ProcessRequest
from msmodelslim.core.graph import AdapterConfig, MappingConfig
from msmodelslim.model.common.layer_wise_forward import generated_decoder_layer_visit_func
from msmodelslim.model.interface_hub import (
    IterSmoothInterface,
    FlexSmoothQuantInterface,
    ModelSlimPipelineInterfaceV1
)
from msmodelslim.model.common.vlm_base import VLMBaseModelAdapter
from msmodelslim.utils.exception import InvalidDatasetError
from msmodelslim.utils.logging import logger_setter, get_logger
from msmodelslim.utils.security import get_valid_read_path, json_safe_load, MAX_READ_FILE_SIZE_32G

from .moe_utils import UnstackedGlm4vMoeTextMoE


@logger_setter()
class GLM4_6VModelAdapter(
    VLMBaseModelAdapter,
    ModelInfoInterface,
    ModelSlimPipelineInterfaceV1,
    IterSmoothInterface,
    FlexSmoothQuantInterface
):
    def __init__(self, model_type: str, model_path: Path, trust_remote_code: bool = False):
        self._processor = None
        super().__init__(model_type, model_path, trust_remote_code)
    
    def _create_model_instance(self, model_cls) -> nn.Module:
        """创建模型实例的基础方法，子类可重写以修改参数（如torch_dtype）"""
        return model_cls.from_pretrained(
            self.model_path,
            config=self.config,
            trust_remote_code=self.trust_remote_code,
            torch_dtype="auto",
            local_files_only=True,
            device_map="cpu",  # All on CPU for now
            attn_implementation='eager'  # Required: prevents KeyError when accessing ALL_ATTENTION_FUNCTIONS
        ).eval()

    def get_model_pedigree(self) -> str:
        """Return model pedigree for best practice matching"""
        return 'glm4_6v'
    
    def get_model_type(self) -> str:
        """Return model type"""
        return self.model_type
    
    def handle_dataset(self, dataset: Any, device: DeviceType = DeviceType.NPU) -> List[Any]:
        """
        Handle multimodal VLM calibration dataset for GLM-4.6V
        
        Supported sample structure (preferred):
            VlmCalibSample(text: str, image: str)
        """
        self._processor = AutoProcessor.from_pretrained(
            self.model_path,
            trust_remote_code=self.trust_remote_code,
            local_files_only=True
        )
        # Validate dataset modality: GLM-4.6V adapter currently requires both image and text for calibration.
        for item in dataset:
            image_path = item.image
            text = item.text
            if image_path is None or text is None:
                raise InvalidDatasetError(
                    (
                        "GLM-4.6V adapter currently requires both image and text "
                        "for calibration."
                    ),
                    action=(
                        "Please use multimodal (image+text) calibration data; pure-text or "
                        "missing image is not supported yet."
                    )
                )

        # Preprocess each sample
        processed_data = []
        for item in tqdm(dataset, desc="Processing calibration dataset"):
            # Support dataclass
            image_path = item.image
            text = item.text

            # Build messages based on presence of image
            # Validate image path
            image_path = get_valid_read_path(image_path)
            content = [
                {"type": "image", "image": str(image_path)},
                {"type": "text", "text": text},
            ]

            messages = [
                {
                    "role": "user",
                    "content": content
                }
            ]
            
            inputs = self._processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt"
            )
            
            processed_item = self._collect_inputs_to_device(
                inputs,
                device,
                keys=[
                    'input_ids',
                    'attention_mask',
                    'position_ids',
                    'past_key_values',
                    'inputs_embeds',
                    'labels',
                    'pixel_values',
                    'pixel_values_videos',
                    'image_grid_thw',
                    'video_grid_thw',
                    'cache_position',
                    'logits_to_keep',
                ],
                defaults={'logits_to_keep': 0}
            )
            
            processed_data.append(processed_item)
        
        get_logger().info(f"Processed {len(processed_data)} multimodal vlm samples")
        return processed_data
    
    def init_model(self, device: DeviceType = DeviceType.NPU) -> nn.Module:
        """
        Initialize model with vision encoder on CPU and text decoder with only 1 layer.
        
        Returns:
            Model with vision encoder + 1 decoder layer loaded, others on meta
        """
        get_logger().info("Initializing GLM-4.6V model with v1 framework (layer-wise loading)...")
        
        # Save original layer count
        origin_layers = self.config.text_config.num_hidden_layers
        get_logger().info(f"Model with {origin_layers} text layers + {self.config.vision_config.depth} vision layers")
        
        # Temporarily set to 1 layer for initialization
        self.config.text_config.num_hidden_layers = 1
        self.config.use_cache = False  # Disable cache to save memory
        
        # Validate model path
        self.model_path = get_valid_read_path(str(self.model_path), is_dir=True, check_user_stat=True)
        
        # Load model with only 1 text decoder layer
        # Vision encoder is fully loaded, text decoder has only 1 layer
        get_logger().info("Loading vision encoder and first text decoder layer...")
        model = self._create_model_instance(Glm4vMoeForConditionalGeneration)

        # Restore original layer count
        self.config.text_config.num_hidden_layers = origin_layers
        
        # Ensure _attn_implementation is set for dynamically loaded layers
        # This prevents KeyError when layers access ALL_ATTENTION_FUNCTIONS[config._attn_implementation]
        self.config.text_config._attn_implementation = 'eager'
        
        # CRITICAL: Copy text_config attention heads to model.config for OV smoothing
        # BaseSmoothProcessor._apply_standard_ov_smooth() reads from model.config, not model.config.text_config
        # This must be done AFTER model is loaded
        if hasattr(model.config.text_config, 'num_attention_heads'):
            model.config.num_attention_heads = model.config.text_config.num_attention_heads
            get_logger().info(f"Set model.config.num_attention_heads = {model.config.num_attention_heads}")
        if hasattr(model.config.text_config, 'num_key_value_heads'):
            model.config.num_key_value_heads = model.config.text_config.num_key_value_heads
            get_logger().info(f"Set model.config.num_key_value_heads = {model.config.num_key_value_heads}")
        
        get_logger().info(f"Model initialized with {origin_layers} layers (1 loaded, others will be loaded on-demand)")
        
        return model

    def generate_model_visit(self, model: nn.Module) -> Generator[ProcessRequest, Any, None]:
        """
        Generate model visit pipeline for layer-wise processing.
        
        Uses the common layer-wise visit function for consistent behavior.
        
        Processing order:
            1. Vision encoder (model.visual) - processed as a whole
            2. Text decoder layers (model.language_model.layers[0..N]) - loaded on-demand
        
        Yields:
            ProcessRequest(name, module, args, kwargs)
        """
        # 1. Process vision encoder first
        get_logger().info("Processing vision encoder...")
        yield ProcessRequest(
            name="model.visual",
            module=model.model.visual,
            args=(),
            kwargs={}
        )
        
        # 2. Process text decoder layers one by one using standard visit function
        get_logger().info("Processing text decoder layers...")
        yield from generated_decoder_layer_visit_func(
            model, 
            transformer_blocks=self.generate_decoder_layer(model)
        )
    
    def generate_model_forward(self, model: nn.Module, inputs: Any) -> Generator[ProcessRequest, Any, None]:
        """
        Generate model forward pipeline for calibration.

        Replicates the call flow in transformers 5.0.0rc0 of Glm4vMoeModel.forward + Glm4vMoeTextModel.forward
        (modeling_glm4v_moe.py L1433-1536, L1050-1133), but yields each submodule
        (vision, then decoder layers) as ProcessRequest for layer-wise calibration.

        Processing order:
            1. Vision encoder -> image features
            2. Merge image features into text embeddings (Glm4vMoeModel L1459-1466)
            3. position_ids via get_rope_index (Glm4vMoeModel L1474-1518, before language_model)
            4. cache_position, 3D/4D position_ids, causal_mask, rotary_emb (Glm4vMoeTextModel L1071-1114)
            5. Decoder layers one by one (Glm4vMoeTextModel L1116-1126)

        Args:
            model:  Glm4vMoeForConditionalGeneration instance
            inputs: Preprocessed data list from handle_dataset

        Yields:
            ProcessRequest(name, module, args, kwargs) for vision then each decoder layer
        """
        # 1. Initialize sample
        sample = inputs
    
        # 2. Extract all inputs (following glm4v_moe.forward signature)
        input_ids = sample.get('input_ids')
        attention_mask = sample.get('attention_mask')
        position_ids = sample.get('position_ids')
        pixel_values = sample.get('pixel_values')
        pixel_values_videos = sample.get('pixel_values_videos')
        image_grid_thw = sample.get('image_grid_thw')
        video_grid_thw = sample.get('video_grid_thw')

        # 3. Run vision encoder to get image features
        # --- Vision: image features (Glm4vMoeModel.forward L1384-1388) ---
        # visual(pixel_values, grid_thw); then split by image and concat
        # for masked_scatter. Source: get_image_features + L1384-1388.
        image_embeds = yield ProcessRequest(
            name="model.visual",
            module=model.model.visual,
            args=(pixel_values, image_grid_thw),
            kwargs={}
        )

        # Split image_embeds per image (same as get_image_features L1386-1387).
        split_sizes = (image_grid_thw.prod(-1) // model.model.visual.spatial_merge_size**2).tolist()
        image_embeds_split = torch.split(image_embeds, split_sizes)

        # 4. Get text token embeddings
        # Text embeddings then merge image features (Glm4vMoeModel L1459-1472).
        # get_placeholder_mask returns masks for image/video token positions; masked_scatter
        # writes image_embeds into those positions so the sequence has [img_tokens, text_tokens].
        inputs_embeds = model.model.get_input_embeddings()(input_ids)
        image_embeds_cat = torch.cat(image_embeds_split, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)
        image_mask, _ = model.model.get_placeholder_mask(input_ids, inputs_embeds, image_features=image_embeds_cat)
        inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds_cat)

        # 5. Build position_ids via get_rope_index
        # --- Glm4vMoeModel.forward: position_ids before calling language_model (L1474-1518) ---
        # get_rope_index computes 3D RoPE indices (temporal/height/width) for vision+text;
        # used for rotary embeddings. attention_mask 4D->2D conversion matches.
        if position_ids is None:
            attention_mask_tensor = (
                attention_mask if not isinstance(attention_mask, dict) else attention_mask["full_attention"]
            )
            if attention_mask_tensor is not None and attention_mask_tensor.ndim == 4:
                attention_mask_tensor = torch.diagonal(attention_mask_tensor[:, 0], dim1=1, dim2=2)
                if attention_mask_tensor.dtype.is_floating_point:
                    attention_mask_tensor = attention_mask_tensor / torch.finfo(attention_mask_tensor.dtype).min
                    attention_mask_tensor = (1.0 - attention_mask_tensor).int()
            position_ids, _ = model.model.get_rope_index(
                input_ids,
                image_grid_thw,
                video_grid_thw,
                attention_mask=attention_mask_tensor,
            )

        # --- Glm4vMoeTextModel.forward (L1050-1133) ---

        # 6. Build cache_position
        # cache_position: indices of current chunk in the full sequence (L1071-1075).
        # past_seen_tokens = past_key_values.get_seq_length() if cache else 0; in calibration
        # past_seen_tokens=0 -> [0, 1, ..., seq_len-1].
        cache_position = torch.arange(0, inputs_embeds.shape[1], device=inputs_embeds.device)

        # position_ids: expand to 3D [3, batch, seq] for temporal/h/w; handle packed 4D case (L1078-1098).
        # If shape is [4, batch, seq], dim 0 is text-only positions for causal mask; 1:4 are 3D RoPE.
        if position_ids.ndim == 2:
            position_ids = position_ids[None, ...].expand(3, position_ids.shape[0], -1)

        if position_ids.ndim == 3 and position_ids.shape[0] == 4:
            text_position_ids = position_ids[0]
            position_ids = position_ids[1:]
        else:
            text_position_ids = None

        # 7. Get causal mask for attention (L1100-1109).
        mask_kwargs = {
            "config": model.config.text_config,
            "input_embeds": inputs_embeds,
            "attention_mask": attention_mask,
            "cache_position": cache_position,
            "past_key_values": None,
            "position_ids": text_position_ids,
        }
        causal_mask = create_causal_mask(**mask_kwargs)

        # 8. Rotary position embeddings, shared by all decoder layers (L1113-1114).
        position_embeddings = model.model.language_model.rotary_emb(inputs_embeds, position_ids)

        # 9. Run decoder layers (L1116-1126). Kwargs match Glm4vMoeTextDecoderLayer.forward (L545-554).
        hidden_states = inputs_embeds
        for name, layer in self.generate_decoder_layer(model):
            hidden_states = yield ProcessRequest(
                name=name,
                module=layer,
                args=(hidden_states,),
                kwargs={
                    'attention_mask': causal_mask,
                    'position_ids': position_ids,
                    'cache_position': cache_position,
                    'position_embeddings': position_embeddings,
                    'past_key_values': None,
                    'use_cache': False,
                }
            )

    def generate_decoder_layer(self, model: nn.Module) -> Generator[Tuple[str, nn.Module], None, None]:
        """
        Generate decoder layers, loading them on-demand from safetensors.

        Yields:
            (layer_name, layer_module) tuples
        """
        num_layers = self.config.text_config.num_hidden_layers
        for layer_idx in range(num_layers):
            name = f"model.language_model.layers.{layer_idx}"
            layer = self._load_decoder_if_not_exist(model, name, layer_idx)
            yield name, layer

    def enable_kv_cache(self, model: nn.Module, need_kv_cache: bool) -> None:
        model.config.use_cache = need_kv_cache
        get_logger().info(f"KV cache {'enabled' if need_kv_cache else 'disabled'}")

    def get_adapter_config_for_subgraph(self) -> List[AdapterConfig]:
        """
        Get adapter config for subgraph-based anti-outlier processing (iter_smooth).
        
        Defines the subgraph structure for norm-linear, ov, and other fusions.
        Based on qwen3vl.py implementation but adapted for Qwen3-VL-MoE model.
        
        Includes both vision encoder and text decoder layers.
        """
        adapter_config = []
        
        # Text decoder layers
        for layer_idx in range(self.config.text_config.num_hidden_layers):
            # Norm-Linear: input_layernorm -> QKV
            text_norm_linear_mapping_config = MappingConfig(
                source=f"model.language_model.layers.{layer_idx}.input_layernorm",
                targets=[
                    f"model.language_model.layers.{layer_idx}.self_attn.q_proj",
                    f"model.language_model.layers.{layer_idx}.self_attn.k_proj",
                    f"model.language_model.layers.{layer_idx}.self_attn.v_proj"
                ]
            )
            
            # OV fusion: V -> O
            text_ov_mapping_config = MappingConfig(
                source=f"model.language_model.layers.{layer_idx}.self_attn.v_proj",
                targets=[f"model.language_model.layers.{layer_idx}.self_attn.o_proj"]
            )

            adapter_config.extend([
                AdapterConfig(
                    subgraph_type="norm-linear",
                    mapping=text_norm_linear_mapping_config
                ),
                AdapterConfig(
                    subgraph_type="ov",
                    mapping=text_ov_mapping_config,
                    extra_config={
                        'group_method': 'max'
                    }
                ),
            ])

            if layer_idx < self.config.text_config.first_k_dense_replace:
                text_norm_mlp_mapping_config = MappingConfig(
                    source=f"model.language_model.layers.{layer_idx}.post_attention_layernorm",
                    targets=[
                        f"model.language_model.layers.{layer_idx}.mlp.gate_proj",
                        f"model.language_model.layers.{layer_idx}.mlp.up_proj",
                    ]
                )

                text_up_down_mapping_config = MappingConfig(
                    source=f"model.language_model.layers.{layer_idx}.mlp.up_proj",
                    targets=[f"model.language_model.layers.{layer_idx}.mlp.down_proj"]
                )

                adapter_config.extend([
                    AdapterConfig(
                        subgraph_type="norm-linear",
                        mapping=text_norm_mlp_mapping_config
                    ),
                    AdapterConfig(
                        subgraph_type="up-down",
                        mapping=text_up_down_mapping_config
                    )
                ])
        
        return adapter_config

    def _load_decoder_if_not_exist(self, model: nn.Module, name: str, idx: int) -> nn.Module:
        """
        Load a specific decoder layer from safetensors if not already loaded.

        Layer structure matches Glm4vMoeTextDecoderLayer (modeling_glm4v_moe L530-576):
        input_layernorm -> self_attn -> residual; post_attention_layernorm -> mlp (MoE or MLP) -> residual.

        Steps:
            1. Return the layer if it is already resident (not on meta device).
            2. Otherwise create a fresh Glm4vMoeTextDecoderLayer, load weights
               from safetensors via _get_state_dict, and attach to model.model.language_model.layers.
            3. If it is a MoE layer (layer_idx >= first_k_dense_replace), convert
               Glm4vMoeTextMoE 3D fused experts to nn.Linear via UnstackedGlm4vMoeTextMoE.

        Args:
            model: The full model instance
            name:  Full path (e.g. "model.language_model.layers.5")
            idx:   Layer index

        Returns:
            Loaded (and potentially converted) decoder layer module
        """
        try:
            decoder = model.get_submodule(name)
            try:
                _ = decoder.input_layernorm.weight.device
                get_logger().debug(f"Layer {idx} already loaded")
                return decoder
            except RuntimeError:
                pass
        except AttributeError:
            pass

        get_logger().info(f"Loading decoder layer {idx}...")

        with patch.object(nn.Linear, 'reset_parameters', lambda _self: None):
            get_logger().info(f"Creating decoder layer {idx} structure...")
            decoder = Glm4vMoeTextDecoderLayer(self.config.text_config, layer_idx=idx)

            # 与Qwen3-VL-MoE不同（权重和线性层定义都是三维的），GLM-4.6V只是线性层的定义是三维的nn.Parameter
            if self._is_moe_layer(idx):
                decoder.mlp = UnstackedGlm4vMoeTextMoE(self.config.text_config, decoder.mlp)
            
            # Use strict=True here. In unstacked MoE layers, mlp.gate.e_score_correction_bias
            # is represented as nn.Parameter, so it is loaded and later saved correctly.
            state_dict = self._get_state_dict(decoder, prefix=name)
            decoder.load_state_dict(state_dict, strict=True)
            decoder.eval()

            module_list: nn.ModuleList = model.model.language_model.layers
            if len(module_list) <= idx:
                module_list.append(decoder)
            else:
                module_list[idx] = decoder

            get_logger().info(f"Decoder layer {idx} loaded successfully")

        return decoder

    def _is_moe_layer(self, layer_idx: int) -> bool:
        """
        Return True if this layer uses Glm4vMoeTextMoE (vs plain Glm4vMoeTextMLP).
        """
        return layer_idx >= self.config.text_config.first_k_dense_replace
    
    @lru_cache(maxsize=1)
    def _get_weight_map(self) -> Dict[str, str]:
        """Get weight map from model.safetensors.index.json"""
        index_path = os.path.join(self.model_path, "model.safetensors.index.json")
        index_data = json_safe_load(index_path)
        return index_data['weight_map']

    def _get_state_dict(self, module: nn.Module, prefix: str = "") -> Dict[str, torch.Tensor]:
        """
        Load state dict for a specific module from safetensors files.
        
        Args:
            module: The module to load weights for
            prefix: Name prefix for the module in the full model
        
        Returns:
            State dict for the module
        """
        weight_map = self._get_weight_map()
        
        # For GLM-4.6V unstacked MoE, e_score_correction_bias is promoted to a Parameter
        param_names = [name for name, _ in module.named_parameters()]
        
        # Group by safetensors file
        file_groups = defaultdict(list)
        for param_name in param_names:
            full_name = f"{prefix}.{param_name}" if prefix else param_name
            if full_name in weight_map:
                file_name = weight_map[full_name]
                file_groups[file_name].append(param_name)
        
        # Load weights file by file
        state_dict = {}
        for file_name, names in tqdm(file_groups.items(), desc=f"Loading {prefix}", leave=False):
            file_path = os.path.join(self.model_path, file_name)
            file_path = get_valid_read_path(file_path, extensions='safetensors', size_max=MAX_READ_FILE_SIZE_32G)
            
            with safe_open(file_path, framework='pt', device='cpu') as f:
                for param_name in names:
                    full_name = f"{prefix}.{param_name}" if prefix else param_name
                    state_dict[param_name] = f.get_tensor(full_name)
        
        return state_dict