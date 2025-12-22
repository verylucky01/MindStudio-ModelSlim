# OSP1_2 推理优化示例

## 模型介绍

[Open-Sora-Plan v1.2](https://github.com/PKU-YuanGroup/Open-Sora-Plan) 是一个开源的多模态视频生成模型，由北大-兔展AIGC联合实验室共同发起，专注于高效视频生成任务。

## 环境配置
请参考 [多模态视图生成推理优化工具](../../docs/zh/feature_guide/scripts_based_quantization_and_other_features/pytorch/inference_optimization_for_multimodal_generative_model.md#环境要求) 完成环境配置。

## 支持的模型版本与优化策略

| 模型系列 | 模型版本 | HuggingFace链接 | 采样优化 | DiT缓存优化 | 优化命令 |
|---------|---------|----------------|---------|-------------|----------|
| **Open-Sora-Plan** | v1.2 | [Open-Sora-Plan v1.2](https://github.com/PKU-YuanGroup/Open-Sora-Plan) | ✅ | ✅ | [采样优化](#采样优化) / [DiT缓存优化](#dit缓存优化) |

**说明：**
- ✅ 表示该优化策略已通过msModelSlim官方验证，功能完整、性能稳定，建议优先采用。
- 空格表示该优化策略暂未通过msModelSlim官方验证，用户可根据实际需求进行配置尝试，但优化效果和功能稳定性无法得到官方保证。
- 点击优化命令列中的链接可跳转到对应的具体优化命令

### 已验证优化方法
| 优化类型 | 支持场景 | 加速效果 | 精度损失 |
|---------|---------|---------|---------|
| 采样优化 | 29帧480p | 2x | <1% |
| DiT缓存优化 | 29帧480p/93帧720p | 1.3x | <1% |


## 使用说明

### <span id="采样优化">采样优化</span>

#### 1. 生成校准视频
使用原始模型生成一批视频，用于后续优化步骤的质量评估基准。
```bash
torchrun --nnodes=1 --nproc_per_node 8  --master_port 29503 \
    -m example.osp1_2.sample_t2v_sp \
    --model_path /path/to/checkpoint-xxx/model_ema \
    --num_frames 29 \
    --height 480 \
    --width 640 \
    --cache_dir "../cache_dir" \
    --text_encoder_name google/mt5-xxl \
    --text_prompt examples/prompt_list_0.txt \
    --ae CausalVAEModel_D4_4x8x8 \
    --ae_path "/path/to/causalvideovae" \
    --save_img_path "./sample_video_test" \
    --fps 24 \
    --guidance_scale 7.5 \
    --num_sampling_steps 100 \
    --enable_tiling \
    --tile_overlap_factor 0.125 \
    --save_memory \
    --max_sequence_length 512 \
    --sample_method EulerAncestralDiscrete \
    --model_type "dit"
```
完整示例脚本: [`generate_baseline_t2v_sp.sh`](generate_baseline_t2v_sp.sh)
- **参数介绍**:

    | 参数 | 说明 |
    |------|------|
    | `--model_path` | 预训练的 DiT 模型权重路径 |
    | `--num_frames` | 生成视频的帧数 |
    | `--height`, `--width` | 生成视频的高度和宽度 |
    | `--cache_dir` | Hugging Face 模型缓存目录 |
    | `--text_encoder_name` | 文本编码器的名称或路径 |
    | `--text_prompt` | 包含文本提示的 txt 文件路径 |
    | `--ae` | 使用的自动编码器模型名称 |
    | `--ae_path` | VAE 模型权重路径 |
    | `--save_img_path` | 生成视频的保存路径 |
    | `--fps` | 生成视频的帧率 |
    | `--guidance_scale` | 文本引导的权重比例 |
    | `--num_sampling_steps` | 原始采样步数 |
    | `--enable_tiling` | 是否启用分块推理 |
    | `--tile_overlap_factor` | 分块推理时的重叠因子 |
    | `--save_memory` | 是否启用节省内存模式 |
    | `--max_sequence_length` | 文本编码器的最大序列长度 |
    | `--sample_method` | 使用的采样器方法 |
    | `--model_type` | 模型类型，默认值："dit" |

#### 2. 搜索优化采样步骤
根据生成的校准视频，搜索最优的采样时间步组合，以在保证质量的同时减少采样步数。
```bash
torchrun --nnodes=1 --nproc_per_node 8  --master_port 29503 \
    -m example.osp1_2.search_t2v_sp \
    --model_path /path/to/checkpoint-xxx/model_ema \
    --num_frames 29 \
    --height 480 \
    --width 640 \
    --cache_dir "../cache_dir" \
    --text_encoder_name google/mt5-xxl \
    --text_prompt examples/prompt_list_0.txt \
    --ae CausalVAEModel_D4_4x8x8 \
    --ae_path "/path/to/causalvideovae" \
    --fps 24 \
    --guidance_scale 7.5 \
    --num_sampling_steps 50 \
    --enable_tiling \
    --tile_overlap_factor 0.125 \
    --save_memory \
    --max_sequence_length 512 \
    --sample_method EulerAncestralDiscrete \
    --model_type "dit" \
    --save_dir "/path/to/save/schedule/timestep/file" \
    --videos_path "/path/of/calibration/videos" \
    --neighbour_type "uniform" \
    --monte_carlo_iters 5
```
完整示例脚本: [`search_t2v_sp.sh`](search_t2v_sp.sh)
- **参数介绍**:
其他参数同生成校准视频步骤，新增以下参数：

    | 参数 | 说明 |
    |------|------|
    | `--num_sampling_steps` | 目标优化的采样步数（例如，从100步优化到50步） |
    | `--save_dir` | 保存搜索到的优化时间步配置文件的目录 |
    | `--videos_path` | 第1步生成的校准视频所在的路径 |
    | `--neighbour_type` | 采样过程中使用的邻域搜索类型，可选值为 "uniform" 或 "random" |
    | `--monte_carlo_iters` | Monte Carlo 采样的迭代次数 |

#### 3. 使用优化配置进行推理
使用第2步搜索到的优化时间步配置文件进行推理，以验证加速效果和生成质量。
```bash
torchrun --nnodes=1 --nproc_per_node 8  --master_port 29503 \
    -m example.osp1_2.sample_t2v_sp \
    --model_path /path/to/checkpoint-xxx/model_ema \
    --num_frames 29 \
    --height 480 \
    --width 640 \
    --cache_dir "../cache_dir" \
    --text_encoder_name google/mt5-xxl \
    --text_prompt examples/prompt_list_0.txt \
    --ae CausalVAEModel_D4_4x8x8 \
    --ae_path "/path/to/causalvideovae" \
    --save_img_path "./sample_video_test" \
    --fps 24 \
    --guidance_scale 7.5 \
    --num_sampling_steps 100 \
    --enable_tiling \
    --tile_overlap_factor 0.125 \
    --save_memory \
    --max_sequence_length 512 \
    --sample_method EulerAncestralDiscrete \
    --model_type "dit" \
    --schedule_timestep "/path/of/schedule/timestep/file.txt"
```
完整示例脚本: [`sample_t2v_sp.sh`](sample_t2v_sp.sh)
- **参数介绍**:
其他参数同生成校准视频步骤，新增以下参数：

    | 参数 | 说明 |
    |------|------|
    | `--schedule_timestep` | 第2步搜索到的优化时间步配置文件的路径 |

### <span id="dit缓存优化">DiT缓存优化</span>

#### 1. 搜索缓存配置
搜索最优的 DiT 缓存配置，包括缓存的起始层、缓存层数、缓存起始时间步和时间步间隔。
```bash
torchrun --nnodes=1 --nproc_per_node 8  --master_port 29503 \
    -m example.osp1_2.search_t2v_sp \
    --model_path /path/to/checkpoint-xxx/model_ema \
    --num_frames 29 \
    --height 480 \
    --width 640 \
    --cache_dir "../cache_dir" \
    --text_encoder_name google/mt5-xxl \
    --text_prompt examples/prompt_list_0.txt \
    --ae CausalVAEModel_D4_4x8x8 \
    --ae_path "/path/to/causalvideovae" \
    --fps 24 \
    --guidance_scale 7.5 \
    --num_sampling_steps 100 \
    --enable_tiling \
    --tile_overlap_factor 0.125 \
    --save_memory \
    --max_sequence_length 512 \
    --sample_method EulerAncestralDiscrete \
    --model_type "dit" \
    --search_type "dit_cache" \
    --cache_ratio 1.3 \
    --cache_save_path /path/to/save/the/searched/config
```
完整示例脚本: [`dit_cache_search_t2v_sp.sh`](dit_cache_search_t2v_sp.sh)
- **参数介绍**:
其他参数同生成校准视频步骤，新增以下参数：

    | 参数 | 说明 |
    |------|------|
    | `--search_type` | 指定搜索类型为 "dit_cache" |
    | `--cache_ratio` | 缓存搜索的加速比目标（例如1.3x） |
    | `--cache_save_path` | 保存搜索到的缓存配置文件的路径 |

#### 2. 使用缓存配置进行推理
使用第1步搜索到的缓存配置文件进行推理，以验证加速效果和生成质量。
```bash
torchrun --nnodes=1 --nproc_per_node 8  --master_port 29503 \
    -m example.osp1_2.sample_t2v_sp \
    --model_path /path/to/checkpoint-xxx/model_ema \
    --num_frames 29 \
    --height 480 \
    --width 640 \
    --cache_dir "../cache_dir" \
    --text_encoder_name google/mt5-xxl \
    --text_prompt examples/prompt_list_0.txt \
    --ae CausalVAEModel_D4_4x8x8 \
    --ae_path "/path/to/causalvideovae" \
    --save_img_path "./sample_video_test" \
    --fps 24 \
    --guidance_scale 7.5 \
    --num_sampling_steps 100 \
    --enable_tiling \
    --tile_overlap_factor 0.125 \
    --save_memory \
    --max_sequence_length 512 \
    --sample_method EulerAncestralDiscrete \
    --model_type "dit" \
    --dit_cache_config "/path/of/cache/config/file"
```
完整示例脚本: [`dit_cache_sample_t2v_sp.sh`](dit_cache_sample_t2v_sp.sh)
- **参数介绍**:
其他参数同生成校准视频步骤，新增以下参数：

    | 参数 | 说明 |
    |------|------|
    | `--dit_cache_config` | 第1步搜索到的缓存配置文件的路径 |
