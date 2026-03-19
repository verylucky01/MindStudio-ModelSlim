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
import glob
from collections import defaultdict
from functools import lru_cache
from typing import List, Any, Generator, Tuple, Dict
from unittest.mock import patch

import torch
from safetensors import safe_open
from safetensors.torch import save_file
from torch import nn
from tqdm import tqdm
from qwen_omni_utils import process_mm_info

from msmodelslim.core.const import DeviceType
from msmodelslim.app.naive_quantization.model_info_interface import ModelInfoInterface
from msmodelslim.core.base.protocol import ProcessRequest
from msmodelslim.core.graph import AdapterConfig, MappingConfig
from msmodelslim.model.common.layer_wise_forward import generated_decoder_layer_visit_func
from msmodelslim.model.interface_hub import (
    IterSmoothInterface,
    FlexSmoothQuantInterface,
    ModelSlimPipelineInterfaceV1,
    QuaRotInterface,
    AscendV1SaveInterface
)
from msmodelslim.model.common.vlm_base import VLMBaseModelAdapter
from msmodelslim.utils.logging import logger_setter, get_logger
from msmodelslim.utils.security import get_valid_read_path, json_safe_load, json_safe_dump, MAX_READ_FILE_SIZE_32G


@logger_setter()
class Qwen25OmniThinkerModelAdapter(
    VLMBaseModelAdapter,
    ModelInfoInterface,
    ModelSlimPipelineInterfaceV1,
    IterSmoothInterface,
    FlexSmoothQuantInterface,
    AscendV1SaveInterface):
    """
    Adapter for Qwen2.5-Omni model.
    Focuses on quantizing the 'thinker' (LLM) part for ASR scenarios.
    """
    def __init__(self, model_type: str, model_path: str, trust_remote_code: bool = False):
        self._processor = None
        self._tokenizer = None
        super().__init__(model_type, model_path, trust_remote_code)

    def get_model_pedigree(self) -> str:
        """Return model pedigree for best practice matching"""
        return 'qwen2_5_omni_thinker'
    
    def get_model_type(self) -> str:
        """Return model type"""
        return self.model_type

    def handle_dataset(
        self,
        dataset: Any,
        device: DeviceType = DeviceType.NPU
    ) -> List[Any]:
        """
        Prepare calibration dataset for Qwen2.5-Omni.
        """
        from transformers import Qwen2_5OmniProcessor
        # 1. Init processor (once)
        self._processor = Qwen2_5OmniProcessor.from_pretrained(
            self.model_path,
            trust_remote_code=self.trust_remote_code,
            local_files_only=True,
        )

        processed_data = []

        # 2. Preprocess each sample
        for item in tqdm(dataset, desc="Processing Qwen2.5-Omni calibration dataset"):

            # -------- normalize input sample --------
            text = item.text
            image_path = item.image
            audio_path = item.audio
            video_path = item.video
            if text is None or image_path is None or audio_path is None or video_path is None:
                get_logger().warning(f"Sample {item} is missing text, image, audio, or video, skipping...")
                continue
            # -------- build multimodal content --------
            if text and image_path is None and audio_path is None and video_path is None:
                content = text
            else:
                content = []
                content.append(
                    {"type": "text", "text": text}
                )
                if image_path:
                    image_path = get_valid_read_path(image_path)
                    content.append(
                        {"type": "image", "image": str(image_path)}
                    )
                if audio_path:
                    audio_path = get_valid_read_path(audio_path)
                    content.append(
                        {"type": "audio", "audio": str(audio_path)}
                    )
                if video_path:
                    video_path = get_valid_read_path(video_path)
                    content.append(
                        {"type": "video", "video": str(video_path)}
                    )

            conversation = [
                {
                    "role": "system",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, "
                                "capable of perceiving auditory and visual inputs, as well as generating text "
                                "and speech."
                            )
                        }
                    ]
                },
                {
                    "role": "user",
                    "content": content
                }
            ]

            # set use audio in video
            USE_AUDIO_IN_VIDEO = True

            # Preparation for batch inference
            text = self._processor.apply_chat_template(
                conversation, add_generation_prompt=True, tokenize=False
            )
            audios, images, videos = process_mm_info(
                conversation, use_audio_in_video=USE_AUDIO_IN_VIDEO
            )
            inputs = self._processor(
                text=text,
                audio=audios,
                images=images,
                videos=videos,
                return_tensors="pt",
                padding=True,
                use_audio_in_video=USE_AUDIO_IN_VIDEO
            )
            
            processed_item = self._collect_inputs_to_device(
                inputs,
                device,
                keys=[
                    "input_ids",
                    "input_features",
                    "pixel_values",
                    "pixel_values_videos",
                    "image_grid_thw",
                    "video_grid_thw",
                    "attention_mask",
                    "feature_attention_mask",
                    "audio_feature_lengths",
                    "position_ids",
                    "past_key_values",
                    "inputs_embeds",
                    "rope_deltas",
                    "labels",
                    "use_cache",
                    "output_attentions",
                    "output_hidden_states",
                    "return_dict",
                    "use_audio_in_video",
                    "cache_position",
                    "video_second_per_grid",
                ],
                defaults={}
            )

            processed_data.append(processed_item)

        if len(processed_data) == 0:
            raise ValueError(
                "No valid multimodal samples found in dataset for Qwen2.5 Omni. "
                "Qwen2.5 Omni currently requires four modalities (text, image, audio, video) for calibration."
            )

        get_logger().info(
            f"Processed {len(processed_data)} Qwen2.5-Omni multimodal samples"
        )

        return processed_data

    def init_model(self, device: DeviceType = DeviceType.NPU) -> nn.Module:
        """
        Initialize Qwen2.5-Omni.
        """
        try:
            from transformers import Qwen2_5OmniThinkerForConditionalGeneration
        except ImportError as e:
            raise ImportError(
                "Please install transformers with Qwen2.5-Omni support.",
                action="pip install transformers==4.57.3"
            ) from e

        get_logger().info("Initializing Qwen2.5-Omni model...")

        thinker_config = self.config.thinker_config
        origin_layers = thinker_config.text_config.num_hidden_layers
        thinker_config.text_config.num_hidden_layers = 1
        self.config.use_cache = False 

        # 1. Load Skeleton
        self.model_path = get_valid_read_path(str(self.model_path), is_dir=True, check_user_stat=True)
        model = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
            self.model_path, config=thinker_config, trust_remote_code=self.trust_remote_code,
            torch_dtype="auto", local_files_only=True, device_map="cpu", 
            attn_implementation="eager", use_safetensors=True,
        ).eval()

        thinker_config.text_config.num_hidden_layers = origin_layers

        state_dict = self._get_state_dict(model, prefix='thinker')
        model.load_state_dict(state_dict)

        get_logger().info("Model initialized.")

        # Config fix
        model.config.num_attention_heads = thinker_config.text_config.num_attention_heads
        model.config.num_key_value_heads = thinker_config.text_config.num_key_value_heads

        return model

    def generate_model_visit(self, model: nn.Module) -> Generator[ProcessRequest, Any, None]:
        """
        Generate model visit pipeline for layer-wise processing.
        
        Uses the common layer-wise visit function for consistent behavior.
        
        Processing order:
            1. Audio encoder (model.visual) - processed as a whole
            2. Visual encoder (model.visual) - processed as a whole
            3. Text decoder layers (model.language_model.layers[0..N]) - loaded on-demand
        
        Yields:
            ProcessRequest(name, module, args, kwargs)
        """
        # 1. Audio Tower
        get_logger().info("Processing audio encoder...")
        yield ProcessRequest(
            name="audio_tower",
            module=model.audio_tower,
            args=(), kwargs={}
        )

        # 2. Visual Encoder (Includes Merger inside)
        get_logger().info("Processing image vision encoder...")
        yield ProcessRequest(
            name="visual",
            module=model.visual,
            args=(), kwargs={}
        )

        # 3. Visual Encoder (Includes Merger inside)
        get_logger().info("Processing video vision encoder...")
        yield ProcessRequest(
            name="visual",
            module=model.visual,
            args=(), kwargs={}
        )

        # 4. Text Decoder
        get_logger().info("Processing text decoder layers...")
        yield from generated_decoder_layer_visit_func(
            model, 
            transformer_blocks=self.generate_decoder_layer(model)
        )

    def generate_model_forward(self, model: nn.Module, inputs: Any) -> Generator[ProcessRequest, Any, None]:
        """
        Generate model forward pipeline for calibration.
        
        This mimics Qwen2_5OmniThinkerForConditionalGeneration.forward() logic:
            1. Extract text embeddings
            2. Run audio encoder to get audio features  
            3. Run vision encoder to get image/video features
            4. Merge multimodal features into text embeddings (masked_scatter)
            5. Compute 3D RoPE position encoding
            6. Run each text decoder layer with proper inputs
            7. Apply final layer norm and lm_head
        
        Args:
            model: The Qwen2.5-Omni thinker model
            inputs: Preprocessed data from handle_dataset
        
        Yields:
            ProcessRequest with forward results
        """
        from transformers.masking_utils import create_causal_mask, create_sliding_window_causal_mask
        # 1. Initialize sample
        sample = inputs
        
        # 2. Extract all inputs (following thinker.forward signature)
        input_ids = sample.get('input_ids')
        input_features = sample.get('input_features')
        pixel_values = sample.get('pixel_values')
        pixel_values_videos = sample.get('pixel_values_videos')
        image_grid_thw = sample.get('image_grid_thw')
        video_grid_thw = sample.get('video_grid_thw')
        attention_mask = sample.get('attention_mask')
        feature_attention_mask = sample.get('feature_attention_mask')
        audio_feature_lengths = sample.get('audio_feature_lengths')
        position_ids = sample.get('position_ids')
        use_audio_in_video = sample.get('use_audio_in_video', False)
        video_second_per_grid = sample.get('video_second_per_grid')
        
        # 3. Get text embeddings
        inputs_embeds = model.get_input_embeddings()(input_ids)
        
        # 4. Process Audio Features
        if input_features is not None:
            # Preprocess input_features following get_audio_features logic
            if feature_attention_mask is not None:
                audio_feature_lengths = torch.sum(feature_attention_mask, dim=1)
                # Permute and filter: (batch, mel_bins, seq_len) -> (mel_bins, total_valid_len)
                input_features = input_features.permute(0, 2, 1)[feature_attention_mask.bool()].permute(1, 0)
            # else: audio_feature_lengths uses value from sample (or None)
            
            # Calculate lengths after CNN (following get_audio_features logic)
            # audio_feature_lengths should be set by now (either from mask or from sample)
            feature_lens_for_calc = audio_feature_lengths if audio_feature_lengths is not None else (
                feature_attention_mask.sum(-1) if feature_attention_mask is not None else None
            )
            if feature_lens_for_calc is None:
                raise ValueError("Either audio_feature_lengths or feature_attention_mask must be provided for audio processing")
            
            audio_feat_lengths, audio_output_lengths = model.audio_tower._get_feat_extract_output_lengths(feature_lens_for_calc)
            feature_lens = audio_feature_lengths if audio_feature_lengths is not None else feature_attention_mask.sum(-1)
            
            # Run audio encoder
            audio_features = yield ProcessRequest(
                name="audio_tower",
                module=model.audio_tower,
                args=(),
                kwargs={
                    'input_features': input_features,
                    'feature_lens': feature_lens,
                    'aftercnn_lens': audio_feat_lengths,
                }
            )
            # Extract last_hidden_state from audio encoder output
            if isinstance(audio_features, dict):
                audio_features = audio_features.get('last_hidden_state', audio_features)
            elif hasattr(audio_features, 'last_hidden_state'):
                audio_features = audio_features.last_hidden_state
            
            # Convert to proper device/dtype
            audio_features = audio_features.to(inputs_embeds.device, inputs_embeds.dtype)
            
            # Get placeholder mask and merge
            _, _, audio_mask = model.get_placeholder_mask(input_ids, inputs_embeds=inputs_embeds)
            inputs_embeds = inputs_embeds.masked_scatter(audio_mask, audio_features)
        
        # 5. Process Image Features
        if pixel_values is not None:
            # Run vision encoder for images
            image_embeds = yield ProcessRequest(
                name="visual",
                module=model.visual,
                args=(pixel_values,),
                kwargs={'grid_thw': image_grid_thw}
            )
            
            # Convert to proper device/dtype
            image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
            
            # Get placeholder mask and merge
            image_mask, _, _ = model.get_placeholder_mask(
                input_ids, inputs_embeds=inputs_embeds, image_features=image_embeds
            )
            inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)
        
        # 6. Process Video Features
        if pixel_values_videos is not None:
            # Run vision encoder for videos
            video_embeds = yield ProcessRequest(
                name="visual",
                module=model.visual,
                args=(pixel_values_videos,),
                kwargs={'grid_thw': video_grid_thw}
            )
            
            # Convert to proper device/dtype
            video_embeds = video_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
            
            # Get placeholder mask and merge
            _, video_mask, _ = model.get_placeholder_mask(
                input_ids, inputs_embeds=inputs_embeds, video_features=video_embeds
            )
            inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)
        
        # 7. Update audio_feature_lengths from mask if needed
        # Note: audio_feature_lengths is already calculated in step 4 if input_features is not None
        # Only calculate here if input_features is None (no audio processing in step 4)
        if input_features is None:
            if feature_attention_mask is not None:
                audio_feature_lengths = torch.sum(feature_attention_mask, dim=1)
            # else: audio_feature_lengths remains as from sample (or None)
        
        # 8. Compute 3D RoPE position IDs
        if attention_mask is not None and position_ids is None:
            # Compute position_ids using get_rope_index
            delta0 = (1 - attention_mask).sum(dim=-1).unsqueeze(1)
            position_ids, rope_deltas = model.get_rope_index(
                input_ids,
                image_grid_thw,
                video_grid_thw,
                attention_mask,
                use_audio_in_video,
                audio_feature_lengths,
                video_second_per_grid,
            )
            rope_deltas = rope_deltas - delta0
            model.rope_deltas = rope_deltas
        
        # Expand position_ids to 3D if needed (for mROPE)
        if position_ids.ndim == 2:
            position_ids = position_ids[None, ...].expand(3, position_ids.shape[0], -1)
        
        # Extract text position ids (for causal mask creation)
        if position_ids.ndim == 3 and position_ids.shape[0] == 4:
            text_position_ids = position_ids[0]
            position_ids = position_ids[1:]
        else:
            text_position_ids = None
        
        # 9. Create cache_position
        cache_position = torch.arange(
            0, inputs_embeds.shape[1], device=inputs_embeds.device
        )
        
        # 10. Create 4D causal attention mask
        # This follows Qwen2_5OmniThinkerTextModel.forward logic
        if not isinstance(attention_mask, dict):
            mask_kwargs = {
                "config": model.model.config,
                "input_embeds": inputs_embeds,
                "attention_mask": attention_mask,
                "cache_position": cache_position,
                "past_key_values": None,
                "position_ids": text_position_ids,
            }
            causal_mask_mapping = {
                "full_attention": create_causal_mask(**mask_kwargs),
            }
            # Add sliding window mask if needed
            if model.model.has_sliding_layers:
                causal_mask_mapping["sliding_attention"] = create_sliding_window_causal_mask(**mask_kwargs)
        else:
            causal_mask_mapping = attention_mask
        
        # 11. Create position embeddings (shared across all layers)
        position_embeddings = model.model.rotary_emb(inputs_embeds, position_ids)
        
        # 12. Process each decoder layer
        hidden_states = inputs_embeds
        for layer_idx, (name, layer) in enumerate(self.generate_decoder_layer(model)):
            # Yield layer result
            hidden_states = yield ProcessRequest(
                name=name,
                module=layer,
                args=(hidden_states,),
                kwargs={
                    'attention_mask': causal_mask_mapping[layer.attention_type],
                    'position_ids': text_position_ids,
                    'position_embeddings': position_embeddings,
                    'past_key_values': None,
                    'output_attentions': False,
                    'use_cache': False,
                    'cache_position': cache_position,
                }
            )
            # Extract hidden states from layer output
            if isinstance(hidden_states, tuple):
                hidden_states = hidden_states[0]
        
    def generate_decoder_layer(self, model: nn.Module) -> Generator[Tuple[str, nn.Module], None, None]:
        """
        Generate Qwen2.5-Omni thinker decoder layers on-demand.

        Decoder structure:
            thinker.model.layers[i]
        """

        text_config = self.config.thinker_config.text_config

        num_layers = text_config.num_hidden_layers

        for layer_idx in range(num_layers):
            layer_name = f"model.layers.{layer_idx}"

            # Load layer if not exists
            layer = self._load_decoder_if_not_exist(
                model=model,
                name=layer_name,
                idx=layer_idx,
            )

            yield layer_name, layer
    
    def enable_kv_cache(self, model: nn.Module, need_kv_cache: bool) -> None:
        """
        Enable/disable KV cache.
        
        For calibration, we typically don't need KV cache.
        """
        model.config.use_cache = need_kv_cache
        get_logger().info(f"KV cache {'enabled' if need_kv_cache else 'disabled'}")

    def get_adapter_config_for_subgraph(self) -> List[AdapterConfig]:
        """
        Get adapter config for subgraph-based anti-outlier processing (iter_smooth).
        
        Defines the subgraph structure for norm-linear, ov, and other fusions.
        
        Includes both audio tower, vision encoder and text decoder layers.
        """
        adapter_config = []
        
        audio_config = self.config.thinker_config.audio_config
        vision_config = self.config.thinker_config.vision_config
        text_config = self.config.thinker_config.text_config

        for layer_idx in range(audio_config.num_hidden_layers):
            audio_attn_norm_linear_mapping_config = MappingConfig(
                source=f"audio_tower.layers.{layer_idx}.self_attn_layer_norm",
                targets=[
                    f"audio_tower.layers.{layer_idx}.self_attn.q_proj",
                    f"audio_tower.layers.{layer_idx}.self_attn.k_proj",
                    f"audio_tower.layers.{layer_idx}.self_attn.v_proj"
                ]
            )

            # Skip OV mapping

            adapter_config.extend([
                AdapterConfig(
                    subgraph_type="norm-linear",
                    mapping=audio_attn_norm_linear_mapping_config
                ),
            ])

        for layer_idx in range(vision_config.depth):
            # Norm-Linear: input_layernorm -> fused QKV
            visual_attn_norm_linear_mapping_config = MappingConfig(
                source=f"visual.blocks.{layer_idx}.norm1",
                targets=[
                    f"visual.blocks.{layer_idx}.attn.q",
                    f"visual.blocks.{layer_idx}.attn.k",
                    f"visual.blocks.{layer_idx}.attn.v"
                ]
            )

            # Note: OV mapping is not applicable with fused QKV
            # The qkv -> proj mapping would need special handling
            # For now, we skip the OV subgraph for visual attention

            visual_mlp_norm_linear_mapping_config = MappingConfig(
                source=f"visual.blocks.{layer_idx}.norm2",
                targets=[
                    f"visual.blocks.{layer_idx}.mlp.gate_proj",
                    f"visual.blocks.{layer_idx}.mlp.up_proj"
                ]
            )

            visual_mlp_up_down_mapping_config = MappingConfig(
                source=f"visual.blocks.{layer_idx}.mlp.up_proj",
                targets=[f"visual.blocks.{layer_idx}.mlp.down_proj"]
            )

            adapter_config.extend([
                AdapterConfig(
                    subgraph_type="norm-linear",
                    mapping=visual_attn_norm_linear_mapping_config
                ),
                # Skip OV mapping for fused QKV
                AdapterConfig(
                    subgraph_type="norm-linear",
                    mapping=visual_mlp_norm_linear_mapping_config
                ),
                AdapterConfig(
                    subgraph_type="up-down",
                    mapping=visual_mlp_up_down_mapping_config
                ),
            ])

        for layer_idx in range(text_config.num_hidden_layers):
            text_attn_norm_linear_mapping_config = MappingConfig(
                source=f"model.layers.{layer_idx}.input_layernorm",
                targets=[
                    f"model.layers.{layer_idx}.self_attn.q_proj",
                    f"model.layers.{layer_idx}.self_attn.k_proj",
                    f"model.layers.{layer_idx}.self_attn.v_proj"
                ]
            )
            
            # Note: OV mapping can cause accuracy degradation, so we skip it

            text_mlp_norm_linear_mapping_config = MappingConfig(
                source=f"model.layers.{layer_idx}.post_attention_layernorm",
                targets=[
                    f"model.layers.{layer_idx}.mlp.gate_proj",
                    f"model.layers.{layer_idx}.mlp.up_proj"
                ]
            )

            text_mlp_up_down_mapping_config = MappingConfig(
                source=f"model.layers.{layer_idx}.mlp.up_proj",
                targets=[f"model.layers.{layer_idx}.mlp.down_proj"]
            )

            adapter_config.extend([
                AdapterConfig(
                    subgraph_type="norm-linear",
                    mapping=text_attn_norm_linear_mapping_config
                ),
                AdapterConfig(
                    subgraph_type="norm-linear",
                    mapping=text_mlp_norm_linear_mapping_config
                ),
                AdapterConfig(
                    subgraph_type="up-down",
                    mapping=text_mlp_up_down_mapping_config
                ),
            ])
        
        return adapter_config

    def ascendv1_save_postprocess(self, model: nn.Module, save_directory: str) -> None:
        """为满足 vLLM Ascend 要求，在描述/索引与权重文件中为键名统一添加 thinker. 前缀。"""
        prefix = "thinker."
        for name in ("quant_model_description.json", "quant_model_weights.safetensors.index.json"):
            path = os.path.join(save_directory, name)
            data = json_safe_load(path)
            target = data.get("weight_map") if data.get("weight_map") is not None else data
            json_safe_dump({f"{prefix}{k}": v for k, v in target.items()}, path, indent=2)
        for path in glob.glob(os.path.join(save_directory, "*.safetensors")):
            with safe_open(path, framework="pt", device="cpu") as f:
                save_file({f"{prefix}{k}": f.get_tensor(k) for k in f.keys()}, path)
    
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
        
        # Get all parameter names for this module
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

    def _load_decoder_if_not_exist(
        self,
        model: nn.Module,
        name: str,
        idx: int
    ) -> nn.Module:
        """
        Load a specific Qwen2.5-Omni decoder layer from safetensors if not already loaded.
        """

        # 1. 已存在且已 materialize → 直接返回
        try:
            decoder = model.get_submodule(name)
            try:
                _ = decoder.input_layernorm.weight.device
                get_logger().debug(f"Decoder layer {idx} already loaded")
                return decoder
            except RuntimeError:
                pass
        except AttributeError:
            pass

        get_logger().info(f"Loading Qwen2.5-Omni decoder layer {idx}...")

        # 2. 禁用 reset_parameters
        with patch.object(nn.Linear, "reset_parameters", lambda _self: None):
            from transformers.models.qwen2_5_omni.modeling_qwen2_5_omni import (
                Qwen2_5OmniDecoderLayer
            )
            # 3. 正确创建 Decoder
            decoder = Qwen2_5OmniDecoderLayer(
                config=model.config.text_config,
                layer_idx=idx,
            )

            # 4. 使用 msmodelslim 既有的 state_dict 机制
            # 注意：prefix 必须是 Omni 的真实权重路径
            weight_prefix = f"thinker.model.layers.{idx}"

            state_dict = self._get_state_dict(
                decoder,
                prefix=weight_prefix
            )

            # 5. load_state_dict
            decoder.load_state_dict(state_dict, strict=True)
            decoder.eval()

            # 6. 注册回模型
            module_list: nn.ModuleList = model.model.layers

            if len(module_list) <= idx:
                module_list.append(decoder)
            else:
                module_list[idx] = decoder

            get_logger().info(
                f"Decoder layer {idx} loaded successfully (Qwen2.5-Omni)"
            )

        return decoder