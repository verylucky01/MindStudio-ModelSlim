# 一键量化配置协议说明

## 简介

一键量化功提供了两种使用方式供用户选择，并根据用户选择的方式来判断使用的量化配置协议。
- 方式1：适用于工具已经支持模型的一键量化且用户无特殊量化诉求场景，可通过指定 `quant_type` 参数，工具在最佳实践库中自动匹配最适合的量化配置进行量化。
- 方式2：适用于工具尚未支持模型的一键量化或用户有特殊量化诉求场景，可通过指定 `config_path` 参数，工具直接使用用户指定的自定义量化配置进行量化。

## 基础配置协议功能介绍

### 功能说明

一键量化配置协议采用分层结构设计思想，通过配置描述量化过程使用的量化服务版本、流水线类型、量化处理方式、保存策略以及量化校准集等信息。yaml结构如下：

- **apiversion**: 用于选择后端量化服务的版本。
- **spec**: 具体的量化服务配置字段。

### 配置参数说明

| 参数           | 可选/必选 | 说明                                                                                                                                                                  | 作用                                |
|--------------|-------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------|-----------------------------------|
| apiversion   | 必选    | 1. 当前支持列表：`"modelslim_v0"`、`"modelslim_v1"`、`"multimodal_vlm_modelslim_v1"`、`"multimodal_sd_modelslim_v1"`。<br> 2. 工具根据此字段选择对应的量化服务后端。<br> 3. 不同版本的量化服务可能有不同的配置字段和参数要求。 | 用于选择后端量化服务的版本，不同的量化服务有着不同的具体配置协议。 |
| spec         | 必选    | 1. **流水线定义**：指定量化处理的流水线类型。<br> 2. **处理器配置**：定义各种量化处理器的参数。<br>3. **保存策略**：指定量化结果的保存方式和格式 <br>4. **数据集配置**：指定校准数据集                            | 具体的量化服务配置字段，包含量化策略、处理流程和保存方式等所有具体参数。                                 |

注意：各协议版本的维护策略如下，请尽量使用推荐的协议版本

| 协议版本         | 维护策略 | 状态  |
|--------------|------|-----|
| modelslim_v0 | 即将废弃 | 不推荐 |
| modelslim_v1 | 逐步完善 | 推荐  |
| multimodal_vlm_modelslim_v1 | 逐步完善 | 推荐  |
| multimodal_sd_modelslim_v1 | 逐步完善 | 推荐  |

### 使用示例

```yaml
apiversion: "modelslim_v1"   # 使用modelslim_v1版本的量化服务
spec:
  runner: "auto"
  process: [ ]
  save: [ ]
  dataset: "mix_calib.jsonl"   # 采用mix_calib.jsonl校准集
```
**说明**：该配置文件描述了使用V1量化服务进行空处理。


## modelslim_v0 量化服务配置详解

### 功能说明

modelslim_v0量化服务主要由Calibrator、AntiOutlier等旧版接口组成，其配置协议（YAML）基本与原有的Python-API接口保持一致，便于从旧版本平滑迁移。

**相关文档**:

- [Calibrator.md](../../python_api/foundation_model_compression_apis/foundation_model_quantization_apis/pytorch_Calibrator.md)
- [AntiOutlier.md](../../python_api/foundation_model_compression_apis/foundation_model_quantization_apis/AntiOutlier.md)

## modelslim_v1 量化服务配置详解

### 功能说明

modelslim_v1是量化工具推出的新一代量化处理框架，目前正在快速演进中。

相较于v0版本，v1有以下优势：

- 算法独立实现，配置自由组合。
- 支持逐层量化，大幅降低资源消耗。
- 不依赖特定版本的CANN。

### 配置字段说明

V1量化服务的具体配置应当位于spec字段下。

#### runner - 量化调度器类型

**作用**: 定义量化处理的调度器类型。
**类型**: `string`。
**默认值**: `"auto"`。

| 可选值 | 说明 | 适用场景 | 特点 |
|--------|------|----------|------|
| auto | 工具自动选择 | 大多数场景 | 根据模型大小、可用内存、设备配置自动选择最优策略 |
| layer_wise | [逐层量化](layer_wise_quantization.md) | 大模型（≥7B） | 内存占用低，可能需要适配 |
| dp_layer_wise | [分布式逐层量化](dp_layer_wise_quantization.md) | 大模型（≥7B）多卡场景 | 多设备并行，显著提升量化效率 |
| model_wise | 非逐层量化 | 小模型（<7B） | 内存占用较高，兼容性好 |

#### dataset - 校准数据集配置

**作用**: 指定校准数据集文件名，将会从lab_calib目录下匹配使用的校准集。
**类型**: `string`。
**默认值**: `"mix_calib.jsonl"`。

| 属性 | 说明 |
|------|------|
| 文件位置 | lab_calib目录下 |
| 文件格式 | JSONL格式 |
| 用途 | 用于激活值量化的校准过程 |

#### process - 处理器配置字段

**作用**: 定义量化处理的处理器列表，按顺序执行每个处理器。

**特点**:

- **列表结构**: process 是处理器列表，包含多个处理器配置，不同的处理器配置以type字段为区分。
- **顺序执行**: 处理器按照在列表中的顺序依次执行。
- **灵活组合**: 可以组合不同类型的处理器实现复杂的量化策略，但并非所有配置组合都能正常运行。组合时应遵循以下原则（若缺乏相关使用经验，请参考本文后续提供的配置示例，或咨询专业人员获取指导）：
  - 先离群值抑制后量化，例如，当结合Iterative Smooth与W8A8量化时，需先进行Iterative Smooth，后进行W8A8量化。
  - 避免对同一个层进行多种量化设置，如果在配置文件中多次定义同一层的量化参数，可能导致运行时报错，或出现不符合预期的量化效果（如精度异常、量化功能失效等）。

##### 支持处理器表

| 处理器               | 配置示例       | 配置字段详解       |
| ------------------ | ------------- | ------------- |
| linear_quant       | [linear_quant 配置示例](linear_quant.md/#yaml配置示例) | [linear_quant 配置字段详解](linear_quant.md/#yaml配置字段详解)
| group       | [group 配置示例](group.md/#yaml配置示例) | [group 配置字段详解](group.md/#yaml配置字段详解)
| SmoothQuant | [SmoothQuant 配置示例](../../algorithms_instruction/Smooth_Quant.md#yaml配置示例) | [配置字段详解](../../algorithms_instruction/Smooth_Quant.md#yaml配置字段详解)
| Iterative Smooth | [Iterative Smooth 配置示例](../../algorithms_instruction/Iterative_Smooth.md#yaml配置示例) | [ 配置字段详解](../../algorithms_instruction/Iterative_Smooth.md#yaml配置字段详解)
| Flex Smooth Quant | [Flex Smooth Quant 配置示例](../../algorithms_instruction/Flex_Smooth_Quant.md#yaml配置示例) | [ 配置字段详解](../../algorithms_instruction/Flex_Smooth_Quant.md#yaml配置字段详解)
| KV Smooth | [KV Smooth 配置示例](../../algorithms_instruction/kv_smooth.md#yaml配置示例) | [KV Smooth 配置字段详解](../../algorithms_instruction/kv_smooth.md#yaml配置字段详解)
| KVCache Quant | [KVCache Quant 配置示例](../../algorithms_instruction/KVCache_quant.md#yaml配置示例) | [KVCache Quant 配置字段详解](../../algorithms_instruction/KVCache_quant.md#yaml配置字段详解)
| FA3 Quant | [FA3 Quant 配置示例](../../algorithms_instruction/FA3_quant.md#yaml配置示例) | [FA3 Quant 配置字段详解](../../algorithms_instruction/FA3_quant.md#yaml配置字段详解)
| Histogram     | [Histogram 配置示例](../../algorithms_instruction/histogram_activation_quantization.md#yaml配置示例) | [ 配置字段详解](../../algorithms_instruction/histogram_activation_quantization.md#yaml配置字段详解)
| SSZ       | [SSZ 配置示例](../../algorithms_instruction/ssz.md#yaml配置示例) | [SSZ 配置字段详解](../../algorithms_instruction/ssz.md#yaml配置字段详解)
| Float Sparse   | [Float Sparse 配置示例](../../algorithms_instruction/float_sparse.md#yaml配置示例) | [Float Sparse 配置字段详解](../../algorithms_instruction/float_sparse.md#yaml配置字段详解)
| QuaRot       | [QuaRot 配置示例](../../algorithms_instruction/QuaRot.md#yaml配置示例) | [QuaRot 配置字段详解](../../algorithms_instruction/QuaRot.md#yaml配置字段详解)
| AutoRound       | [AutoRound 配置示例](../../algorithms_instruction/AutoRound.md#作为processor使用) | [AutoRound 配置字段详解](../../algorithms_instruction/AutoRound.md#yaml配置字段详解)
| LAOS       | [LAOS 配置示例](../../algorithms_instruction/LAOS.md#修改配置文件使用) | [LAOS 配置字段详解](../../algorithms_instruction/LAOS.md#yaml配置字段详解)
| PDMIX       | [PDMIX 配置示例](../../algorithms_instruction/pdmix.md#使用方式) | [PDMIX 配置字段详解](../../algorithms_instruction/pdmix.md#使用方式)

#### save - 保存器配置字段

**作用**: 定义量化结果的保存器列表。

##### 支持保存器表

###### ascendv1_saver

**作用**: 保存为ascendv1格式。

**配置示例**:

```yaml
spec:
  save:
    - type: "ascendv1_saver"
      part_file_size: 4            # 分片文件大小（GB）
```

**字段说明**:

| 字段名 | 作用 | 说明 |
|--------|------|------|
| type | 保存器类型标识 | 固定值"ascendv1_saver"，用于标识这是一个Ascend格式保存器 |
| part_file_size | 分片文件大小 | 分片文件的大小，单位为GB |

### 使用示例

- 稠密模型W8A8动态量化：[dense-w8a8-dynamic-v1.yaml](example%2Fdense-w8a8-dynamic-v1.yaml)
- 稠密模型W8A8静态量化：[dense-w8a8-v1.yaml](example%2Fdense-w8a8-v1.yaml)
- 稀疏模型W8A8混合量化：[moe-w8a8-mix-v1.yaml](example%2Fmoe-w8a8-mix-v1.yaml)

## multimodal_sd_modelslim_v1 量化服务配置详解

### 功能说明

multimodal_vlm_modelslim_v1是专门为多模态视觉语言模型（VLM）设计的量化服务，基于modelslim_v1框架构建。

**核心特性**:

- **多模态VLM支持**：针对图像-文本多模态理解模型优化。
- **逐层处理**：采用逐层量化策略，显著减少大模型量化时的显存消耗。
- **多种数据集格式**：支持图像目录和通过JSON/JSONL格式自定义文本prompt的多模态校准数据集。

**适用模型类型**:

- Qwen3-VL-MoE系列：Qwen3-VL-235B-A22B、Qwen3-VL-30B-A3B等多模态模型
- 其他多模态VLM模型：待后续逐步支持

**配置特点**:

- 支持`dataset`字段配置校准数据集，可以是纯图像目录路径或带JSON/JSONL文件（用于描述每个图像的自定义文本prompt）的图像目录路径
- 支持`default_text`字段配置模型自定义文本prompt
- 默认使用layer_wise逐层量化模式，针对大规模多模态模型优化
- 继承modelslim_v1的所有处理器和保存器配置

### 配置字段说明

multimodal_sd_modelslim_v1量化服务的具体配置位于spec字段下。该服务基于modelslim_v1框架构建，处理器配置字段与modelslim_v1保持一致，以下仅说明有区别或特有的配置字段。

#### runner - 量化调度器类型

当前多模态生成模型考虑到显存占用问题，默认且仅支持layer_wise逐层量化形式。runner默认无需配置，配置为非'layer_wise'字段时，会警告提示并自动转换为layer_wise逐层量化形式。

#### dataset - 校准数据集配置

不支持dataset字段配置校准数据集。由于多模态生成模型的量化校准数据处理方式与大语言模型存在明显差异，会通过识别指定dump_data_dir目录中是否存在名为"calib_data.pth"的校准数据，选择直接加载或自动获取并保存两种方式。详见[dump_config - 校准数据捕获配置](configuration_protocols.md#dump_config---校准数据捕获配置)

#### process - 处理器配置字段

此配置字段与 modelslim_v1 保持一致，参考[modelslim_v1 量化服务配置详解/详细配置字段说明/process - 处理器配置字段](configuration_protocols.md#process---处理器配置字段)

#### save - 保存器配置字段

**作用**: 定义量化结果的保存器列表。

##### 支持保存器表

###### mindie_format_saver

**作用**: 保存为MindIE-SD格式，专为多模态生成模型设计。

**配置示例**:

```yaml
spec:
  save:
    - type: "mindie_format_saver"
      part_file_size: 0            # 分片文件大小（0表示不分片）
```

**字段说明**:

| 字段名 | 作用 | 说明 |
|--------|------|------|
| type | 保存器类型标识 | 固定值"mindie_format_saver"，用于标识这是一个MindIE-SD格式保存器 |
| part_file_size | 分片文件大小 | 分片文件的大小，单位为GB，0表示不分片 |

#### multimodal_sd_config - 多模态生成特有配置字段

**作用**: 多模态生成模型特有的配置参数，包含校准数据捕获和模型加载与推理配置。

##### dump_config - 校准数据捕获配置

**作用**: 配置校准数据的捕获方式和存储路径。

**配置示例**:

```yaml
spec:
  multimodal_sd_config:
    dump_config:
      capture_mode: "args"         # 捕获模式
      dump_data_dir: ""            # 校准数据保存目录，空字符串时自动处理为权重保存路径
```

**字段说明**:

| 字段名 | 作用 | 说明 | 可选值 |
|--------|------|------|--------|
| capture_mode | 数据捕获模式 | 指定如何捕获模型的输入数据 | 当前仅支持"args"，其他模式待后续扩展 |
| dump_data_dir | 校准数据目录 | 指定校准数据的检索和保存路径，空字符串时自动处理为使用权重保存路径。指定路径存在calib_data.pth时，直接加载作为校准数据，不存在时程序自动通过dump机制保存并加载校准数据 | 字符串路径 |

**捕获模式说明**:

- **args**: 捕获位置参数，适用于大多数多模态生成模型

##### model_config - 模型加载与推理配置

**作用**: 配置模型加载与推理时的相关参数，用于自定义模型默认加载和推理参数。model_config中可配置的字段与类型限制以多模态生成模型原始推理工程仓为准。

**配置示例**:

```yaml
spec:
  multimodal_sd_config:
    model_config:
      # wan2.1 prompt
      prompt: "A stylish woman walks down a Tokyo street..."
      # wan2.1 加载与推理相关参数
      offload_model: True
      frame_num: 121
      task: "t2v-14B"
      size: "1280*720"
      sample_steps: 50
      ...
```

**字段说明**:

| 字段名 | 作用 | 说明 | 可选值 |
|--------|------|------|--------|
| prompt | 校准提示词 | 用于生成校准数据的文本描述 | 字符串 |
| offload_model | 模型卸载 | 是否在推理后卸载模型到CPU，值为True时开启 | True/False |
| frame_num | 生成帧数 | 视频生成的帧数 | 大于0的整数 |
| task | 任务类型 | 指定模型任务类型，"t2v-14B"表示14B模型的文本生成视频任务、"t2v-1.3B"表示1.3B模型的文本生成视频任务 | "t2v-14B", "t2v-1.3B" |
| size | 生成尺寸 | 视频或图像的尺寸规格 | "1280\*720", "832\*480" |
| sample_steps | 采样步数 | 扩散模型的采样步数 | 大于0的整数 |

### 使用示例

- Wan2.1模型W8A8动态量化：[wan2_1_w8a8_dynamic.yaml](../../../../lab_practice/wan2_1/wan2_1_w8a8_dynamic.yaml)

## multimodal_vlm_modelslim_v1 量化服务配置详解

### 功能说明

multimodal_sd_modelslim_v1是专门为多模态生成模型（如Wan2.1等）设计的量化服务，基于modelslim_v1框架构建。

**核心特性**:

- **多模态支持**：针对文本到视频等任务的模型优化。
- **动态静态量化**：支持动态、静态激活值量化，适应不同的输入场景。
- **逐层处理**：支持逐层量化，显著减少大模型量化时的显存消耗。
- **校准数据缓存**：支持校准数据的缓存和复用，提高量化效率。

**适用模型类型**:

- Wan2.1：支持文本到视频等任务
- 其他多模态生成模型：待后续逐步支持

**配置特点**:

- 支持`multimodal_sd_config`字段，包含模型特定的配置参数
- 支持`dump_config`配置，用于校准数据的捕获和存储
- 支持`model_config`配置，包含模型加载和推理的相关参数

### 配置字段说明

multimodal_vlm_modelslim_v1量化服务的具体配置位于spec字段下。该服务基于modelslim_v1框架构建，处理器配置字段与modelslim_v1保持一致，以下仅说明有区别或特有的配置字段。

#### runner - 量化调度器类型

当前多模态VLM模型考虑到显存占用问题，默认且仅支持layer_wise逐层量化形式。runner默认无需配置，配置为非'layer_wise'字段时，会警告提示并自动转换为layer_wise逐层量化形式。

#### dataset - 校准数据路径配置

**作用**: 指定校准数据集的路径。

**类型**: `string`

**支持的格式**:

| 格式类型 | 说明 | 示例 |
|---------|------|------|
| 短名称 | lab_calib目录下的数据集名称 | `"calibImages"` |
| 纯图像目录 | 包含图像文件的目录路径（支持相对路径和绝对路径） | `"/path/to/images"` 或 `"./images"` |
| 带JSON/JSONL文件描述自定义文本prompt的图像目录 | 包含图像文件和JSON/JSONL文件（用于描述每个图像的自定义文本prompt）的目录路径（支持相对路径和绝对路径） | `"/path/to/images"` 或 `"./images"` |

**JSON/JSONL文件格式说明**:

JSON文件的整个文件必须是一个合法的JSON值（通常是数组或对象）；示例格式如下：

```json
[
  {"image": "/path/to/image1.jpg", "text": "Describe this image."},
  {"image": "/path/to/image2.jpg", "text": "What is in this picture?"}
]
```

JSONL文件的每行是一个独立、完整的JSON值，不能跨行，不能有逗号或方括号包裹整体；示例格式如下：

```json
{"image": "/path/to/image1.jpg", "text": "Describe this image."}
{"image": "/path/to/image2.jpg", "text": "What is in this picture?"}
```

字段说明：
- `image`: 图像文件路径（必需）
- `text`: 文本prompt，即提示文本（非空字符串）

**示例**:

```yaml
spec:
  dataset: "calibImages"  # 使用lab_calib目录下的calibImages
```

```yaml
spec:
  dataset: "/path/to/images"  # 使用绝对路径的图像目录
```

```yaml
spec:
  dataset: "/path/to/images_with_json"  # 使用绝对路径的图像目录（带JSON/JSONL文件描述自定义文本prompt）
```

#### default_text - 默认文本prompt配置

**作用**: 统一指定所有校准图像的默认文本prompt。
**类型**: `string`
**默认值**: `"Describe this image in detail."`
**限制**：不能使用空字符串作为文本prompt，当dataset字段配置为带JSON/JSONL文件（用于描述每个图像的自定义文本prompt）的图像目录时，此字段失效。

#### save - 保存器配置字段

此配置字段与 modelslim_v1 保持一致，参考[modelslim_v1 量化服务配置详解/详细配置字段说明/save - 保存器配置字段](configuration_protocols.md#save---保存器配置字段)

**推荐配置**:

```yaml
spec:
  save:
    - type: "ascendv1_saver"
      part_file_size: 4  # 大模型建议分片保存
```

### 使用示例

- Qwen3-VL-MoE模型W8A8混合量化：[qwen3_vl_moe_w8a8.yaml](../../../../lab_practice/qwen3_vl_moe/qwen3_vl_moe_w8a8.yaml)