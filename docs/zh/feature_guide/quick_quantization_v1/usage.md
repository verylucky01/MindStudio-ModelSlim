---
toc_depth: 3
---
# 一键量化完整指南

## 目录

- [简介](#简介)
- [使用前准备](#使用前准备)
- [快速开始](#快速开始)
  - [命令格式](#命令格式)
  - [参数说明](#参数说明)
  - [使用示例](#使用示例)
- [高级特性](#高级特性)
  - [逐层量化及分布式逐层量化](#逐层量化及分布式逐层量化)
- [量化配置协议详解](#量化配置协议详解)
  - [配置协议概述](#量化配置协议概述)
  - [modelslim_v1 配置详解](#modelslim_v1-配置详解)
  - [multimodal_sd_modelslim_v1 配置详解](#multimodal_sd_modelslim_v1-配置详解)
  - [multimodal_vlm_modelslim_v1 配置详解](#multimodal_vlm_modelslim_v1-配置详解)
  - [modelslim_v0 配置说明](#modelslim_v0-配置说明)
- [附录](#附录)

## 简介 {#简介}

一键量化功能面向零基础用户，集成热门开源模型量化功能，具备“开箱即用”的特性。本功能支持全局调用量化命令，用户指定必要参数后，即可对目标原始权重执行指定的量化操作。

一键量化提供了两种使用方式：

1. **方式1（推荐）**：适用于工具已经支持且用户无特殊量化诉求的主流模型量化场景，可通过指定 `quant_type` 参数，工具在最佳实践库中自动匹配最适合的[量化配置](#量化配置协议详解)进行量化。
2. **方式2**：适用于模型或模型量化方式未收录最佳实践库或用户有特殊量化诉求场景，可通过指定 `config_path` 参数，工具直接使用用户指定的自定义[量化配置](#量化配置协议详解)进行量化。

## 使用前准备 {#使用前准备}

安装 msModelSlim 工具，详情请参见[《msModelSlim工具安装指南》](../../getting_started/install_guide.md)。

## 快速开始 {#快速开始}

### 命令格式 {#命令格式}

一键量化功能通过命令行方式启动，可以通过如下命令运行：

```bash
msmodelslim quant [ARGS]
```

用户输入命令后，系统将根据指定需求，在最佳实践库中匹配到最佳配置从而实施量化。

**注意事项**：
1. 最佳实践库中的配置文件放在 `msmodelslim/lab_practice` 中。
2. 若最佳实践库中未搜寻到最佳配置，系统则会向用户询问是否采用默认配置，即使用 `msmodelslim/lab_practice/default/default.yaml` 实施量化。
3. 如果需要打印量化运行日志，可通过以下环境变量进行设置：

| 环境变量                  | 解释        | 是否可选 | 范围             |
|-----------------------|-----------|------|----------------|
| MSMODELSLIM_LOG_LEVEL | 打印同级及以上日志 | 可选   | INFO(默认),DEBUG |

### 参数说明 {#参数说明}

| 参数名称              | 解释        | 是否可选              | 范围                                                                                   |
|-------------------|-----------|-------------------|--------------------------------------------------------------------------------------|
| model_path        | 模型路径      | 必选                | 类型：Str                                                                               |
| save_path         | 量化权重保存路径  | 必选                | 类型：Str                                                                               |
| device            | 量化设备      | 可选                | 1. 类型：Str <br>2. 参考值：'npu','npu:0,1,2,3','cpu' <br>3. 默认值为"npu"（单设备）<br>4. 当配置文件启用分布式逐层量化，且指定多个设备时（如：'npu:0,1,2,3'），系统启动DP逐层量化，请确定配置的算法是否支持分布式执行，配置方式及算法支持详见[逐层量化及分布式逐层量化](#逐层量化及分布式逐层量化)|
| model_type        | 模型名称      | 必选                | 1. 类型：Str <br>2. 大小写敏感，请参考[大模型支持矩阵](../../model_support/foundation_model_support_matrix.md)                                               |  |
| config_path       | 指定配置路径    | 与"quant_type"二选一  | 1. 类型：Str <br>2. 配置文件格式为yaml <br>3. 当前只支持最佳实践库中已验证的配置，若自定义配置，msModelSlim不为量化结果负责。配置指导可参考[量化配置协议详解](#量化配置协议详解)。 <br> |
| quant_type        | 量化类型      | 与"config_path"二选一 | w4a8, w4a8c8, w8a8, w8a8s, w8a8c8, w8a16, w16a16s，请参考[大模型支持矩阵](../../model_support/foundation_model_support_matrix.md)                                   |
| trust_remote_code | 是否信任自定义代码 | 可选                | 1. 类型：Bool，默认值：False <br>2. 请确保加载的自定义代码文件的安全性，设置为True有安全风险。                          |
| h, help           | 命令行参数帮助信息 | 可选                |               -            |

### 使用示例 {#使用示例}

#### 示例1：使用量化类型参数（推荐方式）

使用一键量化功能量化 Qwen2.5-7B-Instruct 模型，量化方式采用 w8a8：

```bash
msmodelslim quant \
  --model_path ${MODEL_PATH} \
  --save_path ${SAVE_PATH} \
  --device npu \
  --model_type Qwen2.5-7B-Instruct \
  --quant_type w8a8 \
  --trust_remote_code True
```

其中：
- `${MODEL_PATH}` 为 Qwen2.5-7B-Instruct 原始浮点权重路径
- `${SAVE_PATH}` 为用户自定义的量化权重保存路径

#### 示例2：使用配置文件参数

使用自定义配置文件进行量化：

```bash
msmodelslim quant \
  --model_path ${MODEL_PATH} \
  --save_path ${SAVE_PATH} \
  --device npu \
  --model_type ${MODEL_TYPE} \
  --config_path ${CONFIG_PATH} \
  --trust_remote_code ${TRUST_REMOTE_CODE}
```

#### 示例3：多卡分布式量化

使用4张NPU卡进行分布式量化：

```bash
msmodelslim quant \
  --model_path ${MODEL_PATH} \
  --save_path ${SAVE_PATH} \
  --device npu:0,1,2,3 \
  --model_type ${MODEL_TYPE} \
  --quant_type w8a8 \
  --trust_remote_code True
```

**注意**：在配置DP逐层量化之前，请首先确保配置的算法支持分布式执行，详见[逐层量化及分布式逐层量化](#逐层量化及分布式逐层量化)。

## 高级特性 {#高级特性}

### 逐层量化及分布式逐层量化 {#逐层量化及分布式逐层量化}

#### 简介

逐层量化（Layer-wise Quantization）是 modelslim_v1 量化服务的重要特性。通过逐层处理模型，显著降低内存占用，使得大模型量化成为可能。

在此基础上，**分布式逐层量化（DP Layer-wise Quantization）** 通过在多设备上进行数据并行（Data Parallel）来显著提升量化效率，同时保持逐层量化的低内存优势。

#### 适用场景

| 场景类型 | 场景描述 | 推荐方案 | 说明 |
|----------|----------|----------|------|
| 大模型量化 | 32B 及以上规模的模型 | 逐层量化 | 内存优势明显 |
| 内存受限环境 | NPU 内存不足以加载整网 | 逐层量化 | 大幅降低内存需求 |
| 追求量化速度 | 超大模型或大规模校准集 | 分布式逐层量化 | 多卡并行显著加速 |

#### 工作原理与优势对比

| 特性 | 传统量化 (Model-wise) | 逐层量化 (单卡) | 分布式逐层量化 (多卡 DP) |
|----------|----------|----------|----------|
| **处理方式** | 模型级整网处理 | 单设备分层顺序处理 | 多设备并行分层处理 |
| **内存占用** | 模型大小的 2-3 倍 | **单层大小的 2-3 倍** | **单层大小的 2-3 倍** |
| **量化效率** | 快 (针对小模型) | 较慢 (针对大模型) | **显著提升 (多卡并行)** |
| **适用模型** | 小模型 (<32B) | 大模型 (≥32B) | 超大模型或大规模校准集 |

#### 配置方法

**1. 通过配置文件指定 runner**

可以在 YAML 配置文件中指定 `runner` 类型，逐层量化和分布式逐层量化分别指定为 `layer_wise` 和 `dp_layer_wise`。如果设置为 `auto`（默认），系统会根据设备数量自动选择 `layer_wise` 或 `dp_layer_wise`。

```yaml
apiversion: "modelslim_v1"       # 协议版本
spec:
  runner: "dp_layer_wise"       # 量化调度器类型: DP逐层调度器
  process:
    - type: "linear_quant"
      qconfig:
        act:                    # 激活值量化配置
          scope: "per_tensor"   # 静态量化标识：整个张量共用量化参数
          dtype: "int8"         # 量化数据类型。默认：int8
          symmetric: false      # 是否对称量化。默认：false
          method: "minmax"      # 量化方法。默认：minmax
        weight:                 # 权重量化配置
          scope: "per_channel"  # 权重量化粒度：逐通道量化
          dtype: "int8"         # 量化数据类型。默认：int8
          symmetric: true       # 是否对称量化。默认：true
          method: "minmax"      # 量化方法。默认：minmax
      include: ["*"]            # 包含的层，支持通配符。默认：["*"]
```

**2. 通过命令行参数配置设备**

```bash
# 单卡逐层量化
msmodelslim quant --device npu:0 ...

# 多卡分布式逐层量化（自动启用 DP）
msmodelslim quant --device npu:0,1,2,3 ...
```

#### 注意事项

1.  **分布式算法支持**：使用 `dp_layer_wise` 时，必须确保所有处理器（如 `linear_quant`）和算法（如 `minmax`, `ssz`, `iter_smooth`）均支持分布式执行。
2.  **加速比说明**：多卡加速效果受校准集大小影响。若校准集过小，通信开销可能导致加速效果不明显。
3.  **多模态限制**：分布式逐层量化暂不支持多模态模型，多模态场景请使用单卡 `layer_wise`。

#### 模型适配

逐层量化支持范围参考[大模型支持矩阵](../../model_support/foundation_model_support_matrix.md) 中支持一键量化的模型。
分布式逐层量化继承自逐层量化，因此支持所有逐层量化适配的大语言模型。

**注意**：DP逐层量化暂不支持多模态模型。多模态模型请使用单卡逐层量化（`layer_wise`）。

#### 算法适配

逐层量化（`layer_wise`）已支持 modelslim_v1 架构下的所有算法。

分布式逐层量化（`dp_layer_wise`）目前仅支持以下呈现的算法：

**离群值抑制算法**

| 算法名称 | 处理器类型 | 支持状态 | 说明 |
|----------|----------|---------|------|
| Iterative Smooth | iter_smooth | ✅ 支持 | 完全支持分布式执行 |
| Flex Smooth Quant | flex_smooth | ✅ 支持 | 完全支持分布式执行 |

**量化算法**

| 算法名称 | 量化方法 | 支持状态 | 说明 |
|----------|---------|---------|------|
| MinMax | minmax | ✅ 支持 | 完全支持分布式执行 |
| SSZ | ssz | ✅ 支持 | 完全支持分布式执行 |

## 量化配置协议详解 {#量化配置协议详解}

### 量化配置协议概述 {#量化配置协议概述}

一键量化配置协议采用分层结构设计思想，通过YAML把整条量化流水线抽象成配置：使用的量化服务版本、流水线类型、量化处理方式、保存策略以及量化校准集等。开发者只关心“策略和流程”，无需在 Python 里硬编码这些细节。

#### 基本结构

YAML配置文件的基本结构如下：

```yaml
apiversion: "modelslim_v1"   # 协议版本：用于选择后端量化服务的版本
spec:                         # 具体的量化服务配置字段
  runner: "auto"              # 量化调度器类型。默认：auto
  process: [ ]                # 处理器配置列表：按顺序执行每个处理器
  save: [ ]                   # 保存器配置列表：定义量化结果的保存方式
  dataset: "mix_calib.jsonl"  # 校准数据集配置：将会从lab_calib目录下匹配使用的校准集
```

#### 协议版本说明

| 参数           | 可选/必选 | 说明                                                                                                                                                                  | 作用                                |
|--------------|-------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------|-----------------------------------|
| apiversion   | 必选    | 1. 当前支持列表：`"modelslim_v0"`、`"modelslim_v1"`、`"multimodal_vlm_modelslim_v1"`、`"multimodal_sd_modelslim_v1"`。<br> 2. 工具根据此字段选择对应的量化服务后端。<br> 3. 不同版本的量化服务可能有不同的配置字段和参数要求。 | 用于选择后端量化服务的版本，不同的量化服务有着不同的具体配置协议。 |
| spec         | 必选    | 1. **流水线定义**：指定量化处理的流水线类型。<br> 2. **处理器配置**：定义各种量化处理器的参数。<br>3. **保存策略**：指定量化结果的保存方式和格式 <br>4. **数据集配置**：指定校准数据集                            | 具体的量化服务配置字段，包含量化策略、处理流程和保存方式等所有具体参数。                                 |

**协议版本维护策略**：

| 协议版本         | 维护策略 | 状态  |
|--------------|------|-----|
| modelslim_v0 | 即将废弃 | 不推荐 |
| modelslim_v1 | 逐步完善 | 推荐  |
| multimodal_vlm_modelslim_v1 | 逐步完善 | 推荐  |
| multimodal_sd_modelslim_v1 | 逐步完善 | 推荐  |

### modelslim_v1 配置详解 {#modelslim_v1-配置详解}

#### 功能说明

modelslim_v1是量化工具推出的新一代量化处理框架，目前正在快速演进中。

相较于modelslim_v0版本，modelslim_v1有以下优势：

- 算法独立实现，配置自由组合。
- 支持逐层量化，大幅降低资源消耗。
- 不依赖特定版本的CANN。

#### runner - 量化调度器类型 {#runner---量化调度器类型}

**作用**: 定义量化处理的调度器类型。
**类型**: `string`。
**默认值**: `"auto"`。

| 可选值 | 说明 | 适用场景 | 特点 |
|--------|------|----------|------|
| auto | 工具自动选择 | 大多数场景 | 根据模型大小、可用内存、设备配置自动选择最优策略 |
| layer_wise | 逐层量化 | 大模型（≥32B） | 内存占用低，可能需要适配 |
| dp_layer_wise | 分布式逐层量化 | 大模型（≥32B）多卡场景 | 多设备并行，显著提升量化效率|
| model_wise | 非逐层量化 | 小模型（<32B） | 内存占用较高，兼容性好 |

#### process - 处理器配置字段 {#process---处理器配置字段}

**作用**: 定义量化处理的处理器列表，按顺序执行每个处理器。

**特点**:

- **列表结构**: process 是处理器列表，包含多个处理器配置，不同的处理器配置以type字段为区分。
- **顺序执行**: 处理器按照在列表中的顺序依次执行。
- **灵活组合**: 可以组合不同类型的处理器实现复杂的量化策略，但并非所有配置组合都能正常运行。组合时应遵循以下原则（若缺乏相关使用经验，请参考本文后续提供的配置示例，或咨询专业人员获取指导）：
  - 先离群值抑制后量化，例如，当结合Iterative Smooth与W8A8量化时，需先进行Iterative Smooth，后进行W8A8量化。
  - 避免对同一个层进行多种量化设置，如果在配置文件中多次定义同一层的量化参数，可能导致运行时报错，或出现不符合预期的量化效果（如精度异常、量化功能失效等）。

##### 支持处理器表

| 处理器 | 处理器类型 | 配置示例 | 配置字段详解 |
| :--- | :--- | :--- | :--- |
| SmoothQuant | 离群值抑制 | [SmoothQuant 配置示例](../../quantization_algorithms/outlier_suppression_algorithms/smooth_quant.md#yaml配置示例) | [配置字段详解](../../quantization_algorithms/outlier_suppression_algorithms/smooth_quant.md#yaml配置字段详解) |
| Iterative Smooth | 离群值抑制 | [Iterative Smooth 配置示例](../../quantization_algorithms/outlier_suppression_algorithms/iterative_smooth.md#yaml配置示例) | [配置字段详解](../../quantization_algorithms/outlier_suppression_algorithms/iterative_smooth.md#yaml配置字段详解) |
| Flex Smooth Quant | 离群值抑制 | [Flex Smooth Quant 配置示例](../../quantization_algorithms/outlier_suppression_algorithms/flex_smooth_quant.md#yaml配置示例) | [配置字段详解](../../quantization_algorithms/outlier_suppression_algorithms/flex_smooth_quant.md#yaml配置字段详解) |
| Flex AWQ SSZ | 离群值抑制 | [Flex AWQ SSZ 配置示例](../../quantization_algorithms/outlier_suppression_algorithms/flex_awq_ssz.md#yaml配置示例) | [配置字段详解](../../quantization_algorithms/outlier_suppression_algorithms/flex_awq_ssz.md#yaml配置字段详解) |
| KV Smooth | 离群值抑制 | [KV Smooth 配置示例](../../quantization_algorithms/outlier_suppression_algorithms/kv_smooth.md#yaml配置示例) | [KV Smooth 配置字段详解](../../quantization_algorithms/outlier_suppression_algorithms/kv_smooth.md#yaml配置字段详解) |
| QuaRot | 离群值抑制 | [QuaRot 配置示例](../../quantization_algorithms/outlier_suppression_algorithms/quarot.md#yaml配置示例) | [QuaRot 配置字段详解](../../quantization_algorithms/outlier_suppression_algorithms/quarot.md#yaml配置字段详解) |
| linear_quant | 量化 | [线性量化配置示例](../../quantization_algorithms/quantization_algorithms/linear_quant.md#yaml配置示例) | [线性量化配置字段详解](../../quantization_algorithms/quantization_algorithms/linear_quant.md#yaml配置字段详解) |
| group | 量化 | [group 配置示例](group.md#yaml配置示例) | [group 配置字段详解](group.md#yaml配置字段详解) |
| KVCache Quant | 量化 | [KVCache Quant 配置示例](../../quantization_algorithms/quantization_algorithms/kvcache_quant.md#yaml配置示例) | [KVCache Quant 配置字段详解](../../quantization_algorithms/quantization_algorithms/kvcache_quant.md#yaml配置字段详解) |
| FA3 Quant | 量化 | [FA3 Quant 配置示例](../../quantization_algorithms/quantization_algorithms/fa3_quant.md#yaml配置示例) | [FA3 Quant 配置字段详解](../../quantization_algorithms/quantization_algorithms/fa3_quant.md#yaml配置字段详解) |
| Float Sparse | 量化 | [Float Sparse 配置示例](../../quantization_algorithms/quantization_algorithms/float_sparse.md#yaml配置示例) | [Float Sparse 配置字段详解](../../quantization_algorithms/quantization_algorithms/float_sparse.md#yaml配置字段详解) |
| AutoRound | 量化 | [AutoRound 配置示例](../../quantization_algorithms/quantization_algorithms/autoround.md#yaml配置示例) | [AutoRound 配置字段详解](../../quantization_algorithms/quantization_algorithms/autoround.md#yaml配置字段详解) |
| LAOS (W4A4方案) | 综合方案 | [LAOS 配置示例](../../quantization_algorithms/quantization_algorithms/laos.md#yaml配置示例) | [LAOS 配置字段详解](../../quantization_algorithms/quantization_algorithms/laos.md#yaml配置字段详解) |

#### save - 保存器配置字段 {#save---保存器配置字段-v1}

**作用**: 定义量化结果的保存器列表。

##### ascendv1_saver

**作用**: 保存为ascendv1格式。

**配置示例**:

```yaml
spec:
  save:
    - type: "ascendv1_saver"        # 保存器类型：保存为ascendv1格式
      part_file_size: 4            # 分片文件大小（GB）
```

**字段说明**:

| 字段名 | 作用 | 说明 |
|--------|------|------|
| type | 保存器类型标识 | 固定值"ascendv1_saver"，用于标识这是一个Ascend格式保存器 |
| part_file_size | 分片文件大小 | 分片文件的大小，单位为GB |

#### dataset - 校准数据集配置

**作用**: 指定校准数据集文件名，将会从lab_calib目录下匹配使用的校准集。
**类型**: `string`。
**默认值**: `"mix_calib.jsonl"`。

| 属性 | 说明 |
|------|------|
| 文件位置 | lab_calib目录下 |
| 文件格式 | JSONL格式 |
| 用途 | 用于激活值量化的校准过程 |

#### 使用示例

在一键量化中，通过 `qconfig.act.scope` 字段来区分 **静态量化** 与 **动态量化**：
- **静态量化 (`per_tensor`)**：在校准阶段统计并固定量化参数，推理时不再计算。特点是**推理性能最优**，硬件兼容性好。
- **动态量化 (`per_token`)**：在推理时为每个 token 动态计算量化参数。特点是**精度更高**，能有效应对激活值中的离群值分布。
以下示例展示了一份稠密模型的静态量化配置：

```yaml
apiversion: modelslim_v1       # 协议版本

spec:                          # 规格定义
  process:                     # 处理器执行列表
    - type: "linear_quant"     # 处理器类型：线性层量化
      qconfig:
        act:                   # 激活值量化配置
          scope: "per_tensor"  # 静态量化标识：整个张量共用量化参数
          dtype: "int8"        # 数据类型：int8
          symmetric: False     # 非对称量化：False（静态量化常用）
          method: "minmax"     # 量化方法：minmax
        weight:                # 权重量化配置
          scope: "per_channel" # 权重量化粒度：逐通道量化
          dtype: "int8"        # 数据类型：int8
          symmetric: True      # 对称量化：True
          method: "minmax"     # 量化方法：minmax
      include: ["*"]           # 包含的层，支持通配符。默认：["*"]
      exclude: ["*down_proj*"] # 排除的层，支持通配符。默认：[]

  save:                        # 保存器配置列表
    - type: "ascendv1_saver"   # 使用标准 ascendv1 格式保存
      part_file_size: 4        # 权重分片大小：4GB（建议大模型开启）
```

以下示例展示了如何针对MOE模型不同层使用不同的量化策略（混合量化）：

```yaml
apiversion: modelslim_v1       # 协议版本
                               #
# 定义 W8A8 动态量化配置模板
default_w8a8_dynamic: &default_w8a8_dynamic
  act:                         # 激活值配置
    scope: "per_token"         # 动态量化标识：每个 token 独立量化参数
    dtype: "int8"              # 数据类型：int8
    symmetric: True            # 对称量化：True
    method: "minmax"           # 量化算法：minmax
  weight:                      # 权重量化配置
    scope: "per_channel"       # 权重量化粒度：逐通道量化
    dtype: "int8"              # 数据类型：int8
    symmetric: True            # 对称量化：True
    method: "minmax"           # 量化算法：minmax
                               #
# 定义 W8A8 静态量化配置模板
default_w8a8: &default_w8a8    # 静态量化模板定义
  act:                         # 激活值配置
    scope: "per_tensor"        # 静态量化标识：整个张量共用量化参数
    dtype: "int8"              # 数据类型：int8
    symmetric: False           # 非对称量化：False（静态量化常用）
    method: "minmax"           # 量化算法：minmax
  weight:                      # 权重量化配置
    scope: "per_channel"       # 权重量化粒度：逐通道量化
    dtype: "int8"              # 数据类型：int8
    symmetric: True            # 对称量化：True
    method: "minmax"           # 量化算法：minmax
                               #
spec:                          # 规格定义
  process:                     # 处理器执行列表
    - type: "group"            # 使用组合处理器，支持对不同层应用不同配置
      configs:                 # 组合内的子处理器配置列表
        - type: "linear_quant" # 线性层量化子处理器 1
          qconfig: *default_w8a8 # 引用静态量化模板：对 Attention 层使用静态量化以优化性能
          include: ["*self_attn*"] # 匹配包含 self_attn 的层
                               #
        - type: "linear_quant" # 线性层量化子处理器 2
          qconfig: *default_w8a8_dynamic # 引用动态量化模板：对 MLP 层使用动态量化以保障精度
          include: ["*mlp*"]   # 匹配包含 mlp 的层
          exclude: ["*gate"]   # 从上述匹配中排除门控层
                               #
  save:                        # 保存器配置列表
    - type: "ascendv1_saver"   # 使用标准 ascendv1 格式保存
      part_file_size: 4        # 权重分片大小：4GB（建议大模型开启）
```

### multimodal_sd_modelslim_v1 配置详解 {#multimodal_sd_modelslim_v1-配置详解}

#### 功能说明

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

#### runner - 量化调度器类型

当前多模态生成模型考虑到显存占用问题，默认且仅支持layer_wise（逐层量化）形式。runner默认无需配置，配置为非'layer_wise'值时，会警告提示并自动转换为layer_wise（逐层量化）形式。

#### process - 处理器配置字段

此配置字段与 modelslim_v1 保持一致，参考[modelslim_v1 配置详解/process - 处理器配置字段](#process---处理器配置字段)

#### save - 保存器配置字段 {#save---保存器配置字段-sd}

**作用**: 定义量化结果的保存器列表。

##### mindie_format_saver

**作用**: 保存为MindIE-SD格式，专为多模态生成模型设计。

**配置示例**:

```yaml
spec:
  save:
    - type: "mindie_format_saver"   # 保存器类型：保存为MindIE-SD格式
      part_file_size: 0            # 分片文件大小（GB），0表示不分片
```

**字段说明**:

| 字段名 | 作用 | 说明 |
|--------|------|------|
| type | 保存器类型标识 | 固定值"mindie_format_saver"，用于标识这是一个MindIE-SD格式保存器 |
| part_file_size | 分片文件大小 | 分片文件的大小，单位为GB，0表示不分片 |

#### multimodal_sd_config - 多模态生成特有配置字段

**作用**: 多模态生成模型特有的配置参数，包含校准数据捕获和模型加载与推理配置。

##### dump_config - 校准数据捕获配置 {#dump_config---校准数据捕获配置}

**作用**: 配置校准数据的捕获方式和存储路径。

**配置示例**:

```yaml
spec:
  multimodal_sd_config:
    dump_config:
      capture_mode: "args"         # 数据捕获模式。当前仅支持"args"
      dump_data_dir: ""            # 校准数据保存目录。空字符串时自动处理为权重保存路径
```

**字段说明**:

| 字段名 | 作用 | 说明                                                                                                            | 可选值 |
|--------|------|---------------------------------------------------------------------------------------------------------------|--------|
| capture_mode | 数据捕获模式 | 指定如何捕获模型的输入数据                                                                                                 | 当前仅支持"args"，其他模式待后续扩展 |
| dump_data_dir | 校准数据目录 | 指定校准数据的检索和保存路径，空字符串时自动处理为使用权重保存路径。指定路径存在calib_data.pth时，直接加载作为校准数据，当calib_data.pth文件不存在时程序自动通过dump机制保存并加载校准数据 | 字符串路径 |

**捕获模式说明**:

- **args**: 捕获位置参数，适用于大多数多模态生成模型

##### model_config - 模型加载与推理配置

**作用**: 配置模型加载与推理时的相关参数，用于自定义模型默认加载和推理参数。model_config中可配置的字段与类型限制以多模态生成模型原始推理工程仓为准。

**配置示例**:

```yaml
spec:
  multimodal_sd_config:
    model_config:
      prompt: "A stylish woman walks down a Tokyo street..." # 校准提示词
      offload_model: True          # 是否在推理后卸载模型到CPU
      frame_num: 121               # 视频生成的帧数
      task: "t2v-14B"              # 任务类型
      size: "1280*720"             # 生成尺寸规格
      sample_steps: 50             # 采样步数
      ......
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

#### dataset - 校准数据集配置

不支持通过dataset字段配置校准数据集。由于多模态生成模型的量化校准数据处理方式与大语言模型存在明显差异，将通过识别指定 dump_data_dir 目录中是否存在名为"calib_data.pth"的校准数据，选择直接加载或自动获取并保存两种方式。详见[dump_config - 校准数据捕获配置](#dump_config---校准数据捕获配置)

#### 使用示例

- Wan2.1模型W8A8动态量化：[wan2_1_w8a8_dynamic.yaml](https://gitcode.com/Ascend/msmodelslim/blob/master/msmodelslim/lab_practice/wan2_1/wan2_1_w8a8_dynamic.yaml)

### multimodal_vlm_modelslim_v1 配置详解 {#multimodal_vlm_modelslim_v1-配置详解}

#### 功能说明

multimodal_vlm_modelslim_v1是专门为多模态视觉语言模型（VLM）设计的量化服务，基于modelslim_v1框架构建。

**核心特性**:

- **多模态VLM支持**：针对图像-文本多模态理解模型优化。
- **逐层处理**：采用逐层量化策略，显著减少大模型量化时的显存消耗。
- **多种数据集格式**：支持图像目录和通过JSON/JSONL格式自定义文本prompt的多模态校准数据集。

**适用模型类型**:

- Qwen3-VL-MoE系列：Qwen3-VL-235B-A22B、Qwen3-VL-30B-A3B等多模态模型
- 其他多模态VLM模型：待后续逐步支持

**配置特点**:

- 支持`dataset`字段配置校准数据集，可以是仅包含图像目录路径或包含JSON/JSONL文件（用于描述每个图像的自定义文本prompt）的图像目录路径
- 支持`default_text`字段配置模型自定义文本prompt
- 默认使用layer_wise（逐层量化）模式，针对大规模多模态模型优化
- 继承modelslim_v1的所有处理器和保存器配置

#### runner - 量化调度器类型

当前多模态VLM模型考虑到显存占用问题，默认且仅支持layer_wise（逐层量化）形式。runner默认无需配置，配置为非'layer_wise'字段时，会警告提示并自动转换为layer_wise（逐层量化）形式。

#### process - 处理器配置字段 {#process---处理器配置字段-vlm}

此配置字段与 modelslim_v1 保持一致，参考[modelslim_v1 配置详解/process - 处理器配置字段](#process---处理器配置字段)

#### default_text - 默认文本prompt配置 {#default_text---默认文本prompt配置}

**作用**: 统一指定所有校准图像的默认文本prompt。
**类型**: `string`
**默认值**: `"Describe this image in detail."`
**限制**：不能使用空字符串作为文本prompt，当dataset字段配置为包含JSON/JSONL文件（用于描述每个图像的自定义文本prompt）的图像目录时，此字段失效。

#### save - 保存器配置字段 {#save---保存器配置字段-vlm}

此配置字段与 modelslim_v1 保持一致，参考[modelslim_v1 配置详解/save - 保存器配置字段](#save---保存器配置字段-v1)

**推荐配置**:

```yaml
spec:
  save:
    - type: "ascendv1_saver"    # 保存器类型：保存为ascendv1格式
      part_file_size: 4        # 分片文件大小（GB）。建议大模型进行分片保存
```

#### dataset - 校准数据路径配置

**作用**: 指定校准数据集的路径。

**类型**: `string`

**支持的格式**:

| 格式类型                             | 说明 | 示例 |
|----------------------------------|------|------|
| 短名称                              | lab_calib目录下的数据集名称 | `"calibImages"` |
| 纯图像目录                            | 包含图像文件的目录路径（支持相对路径和绝对路径） | `"/path/to/images"` 或 `"./images"` |
| 包含JSON/JSONL文件描述自定义文本prompt的图像目录 | 包含图像文件和JSON/JSONL文件（用于描述每个图像的自定义文本prompt）的目录路径（支持相对路径和绝对路径） | `"/path/to/images"` 或 `"./images"` |

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
  dataset: "calibImages"       # 使用lab_calib目录下的短名称数据集
```
...
```yaml
spec:
  dataset: "/path/to/images"   # 使用绝对路径的图像目录
```
...
```yaml
spec:
  dataset: "/path/to/images_with_json"  # 使用包含JSON/JSONL描述文件的图像目录
```

#### 使用示例

- Qwen3-VL-MoE模型W8A8混合量化：[qwen3_vl_moe_w8a8.yaml](https://gitcode.com/Ascend/msmodelslim/blob/master/msmodelslim/lab_practice/qwen3_vl_moe/qwen3_vl_moe_w8a8.yaml)

### modelslim_v0 配置说明 {#modelslim_v0-配置说明}

#### 功能说明

modelslim_v0量化服务主要由Calibrator、AntiOutlier等旧版接口组成，其配置协议（YAML）基本与原有的Python-API接口保持一致，便于从旧版本平滑迁移。

**相关文档**:

- [Calibrator.md](../../python_api_v0/foundation_model_compression_apis/foundation_model_quantization_apis/pytorch_Calibrator.md)
- [AntiOutlier.md](../../python_api_v0/foundation_model_compression_apis/foundation_model_quantization_apis/AntiOutlier.md)

**注意**：modelslim_v0协议版本即将废弃，不推荐使用。建议使用modelslim_v1或更新的协议版本。

## 附录 {#附录}

### 相关资料

- 对于过大的模型，可以参考[逐层量化及分布式逐层量化](#逐层量化及分布式逐层量化)使用逐层量化，能够明显降低显存使用。
- 对于一键量化支持的多种算法，可以参考[一键量化V1架构支持的算法](../../quantization_algorithms/README.md)。

### 常见问题

#### Q1: 如何选择合适的量化调度器？

- **auto**：适用于大多数场景，工具会自动根据模型大小、可用内存、设备配置选择最优策略
- **layer_wise**：适用于32B及以上规模的模型，内存受限环境
- **dp_layer_wise**：适用于超大模型或大规模校准集的多卡场景
- **model_wise**：适用于小模型（<32B），兼容性最好

#### Q2: 逐层量化与分布式逐层量化的区别？

- **逐层量化**：单设备逐层处理，内存占用为单层大小的 2-3 倍。
- **分布式逐层量化**：多设备并行逐层处理，在保持低内存优势的同时显著提升量化速度。详见[逐层量化及分布式逐层量化](#逐层量化及分布式逐层量化)。

#### Q3: 如何判断是否需要使用逐层量化？

如果遇到以下情况，建议使用逐层量化：
- 模型规模 > 32B
- NPU 内存不足
- 量化过程中出现内存溢出错误

#### Q4: 多卡量化一定能加速吗？

不一定。多卡量化的加速效果受多种因素影响：
- 校准集大小：校准集较小时，通信开销可能超过并行收益。
- 卡数选择：并非卡数越多速度越快，需要根据实际情况合理选择。
- 算法支持：只有支持分布式执行的算法才能在多卡环境下正常工作。详见[逐层量化及分布式逐层量化](#逐层量化及分布式逐层量化)。

#### Q5: 如何选择合适的离群值抑制算法？

- **Iterative Smooth**：首选方案。运行快，精度较高，适用于绝大多数 W8A8 场景。
- **Flex Smooth Quant**：进阶方案。支持自动参数搜索，当 Iterative Smooth 效果不佳时尝试。
- **Flex AWQ SSZ**：低比特必备。专为 INT4/W4A8 等场景设计，使用真实量化器评估误差，精度提升显著。
- **QuaRot**：旋转量化。通过数学旋转消除离群点，可与其他平滑算法叠加，进一步挖掘精度潜力。

#### Q6: 如何选择合适的量化算法？

- **MinMax**：简单高效。INT8 场景优先推荐，在保证精度的前提下速度最快。
- **SSZ**：迭代搜索。INT4/W4A8 等低比特场景优先推荐，通过最小化量化误差提升精度。
- **AutoRound**：高精度上限。适用于对精度极度敏感的场景，虽然运行较慢，但量化效果最接近浮点。
- **PDMIX**：平衡策略。旨在平衡推理过程中的激活量化精度与计算性能。
