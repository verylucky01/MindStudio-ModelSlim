# 自动调优配置协议说明

## 概述

### 配置协议简介

自动调优配置协议采用分层结构设计，顶层包含两个核心字段。

- **strategy**: 调优策略配置，定义量化配置生成策略和量化基础配置。
- **evaluation**: 评估服务配置，定义模型精度评估的方式和量化模型服务化拉起相关参数。

### 配置文件位置

用户需要自定义调优配置文件，可以参考 `docs/zh/feature_guide/auto_tuning/example` 目录下的配置文件格式进行自定义。

## 自动调优基础配置协议

### 基础配置结构

```yaml
strategy:
  type: <strategy_type>  # 调优策略类型，如 standing_high
  # 策略特有配置，不同策略类型有不同的配置字段
  # 详细配置请参考对应算法的文档，如 [Standing High 调优算法](../../algorithms_instruction/standing_high.md)
  template:
    # 量化基础配置，参考一键量化配置协议
  metadata:
    config_id: <config_id>
    label:
      w_bit: 8
      a_bit: 8
      is_sparse: false
      kv_cache: false

evaluation:
  type: service_oriented
  demand:
    expectations:
      - dataset: gsm8k
        target: 83  # 目标精度，单位为百分比，83 表示 83%
        tolerance: 2  # 容差，单位为百分比，2 表示 ±2%
  evaluation:
    type: aisbench
    precheck:  # 可选，正式评估前的预检查配置列表
      # 预检查配置列表，每个元素包含 type 字段
    # 其他测评工具配置
  inference_engine:
    # 推理引擎配置
```

## 配置字段详解

### strategy - 调优策略配置

**作用**: 定义调优策略的类型、参数和量化基础配置。

#### type - 策略类型

**作用**: 指定调优算法的类型，不同的策略类型对应不同的调优算法。

**类型**: `string`

**可选值**: 根据已实现的调优策略而定，例如 `standing_high`。详细的算法说明请参考[算法说明](../../algorithms_instruction/)目录下的相关文档，如 [Standing High 调优算法](../../algorithms_instruction/standing_high.md)。

#### 策略特有配置字段

不同的调优策略类型可能有不同的特有配置字段。例如，`standing_high` 策略包含 `anti_outlier_strategies` 字段用于配置离群值抑制策略。详细的策略特有配置说明请参考对应算法的文档，如 [Standing High 调优算法](../../algorithms_instruction/standing_high.md)。

#### template - 量化基础配置

**作用**: 定义量化处理的基础配置，包括量化调度器、处理器、保存器和数据集配置。该配置是开启调优的起点，基础配置的选择一定程度上会影响调优的迭代次数。

**配置协议**: template 字段的配置协议与一键量化配置协议中的 `spec` 字段保持一致，详细配置说明请参考[一键量化配置协议说明](../quick_quantization/configuration_protocols.md)。

**核心字段**:

- **runner**: 量化调度器类型，定义量化处理的调度方式（auto、layer_wise、dp_layer_wise、model_wise等）
- **process**: 处理器列表，定义量化处理的处理器配置（linear_quant、Iterative Smooth等）
- **save**: 保存器列表，定义量化结果的保存方式（ascendv1_saver等）
- **dataset**: 校准数据集配置，指定校准数据集文件名

**配置示例**:

```yaml
template:
  runner: auto
  process:
    - type: linear_quant
      qconfig:
        act:
          scope: per_tensor
          dtype: int8
          symmetric: false
          method: minmax
        weight:
          scope: per_channel
          dtype: int8
          symmetric: true
          method: minmax
        include: [ "*" ]
        exclude: [ ]
  save:
    - type: ascendv1_saver
      part_file_size: 4
  dataset: mix_calib.jsonl
```

#### metadata - 策略元数据

**作用**: 定义策略的元数据信息，用于标识和分类量化配置。

**字段说明**:

| 字段名 | 作用 | 类型 | 说明 |
|--------|------|------|------|
| config_id | 配置ID | string | 量化配置的标识符 |
| label | 标签 | object | 量化配置的标签信息，包括量化位数、稀疏性等 |

**label 字段说明**:

| 字段名 | 作用 | 类型 | 说明 |
|--------|------|------|------|
| w_bit | 权重量化位数 | int | 权重量化的位数，如8表示8bit量化 |
| a_bit | 激活值量化位数 | int | 激活值量化的位数，如8表示8bit量化 |
| is_sparse | 是否稀疏 | bool | 是否为稀疏量化 |
| kv_cache | 是否量化KV缓存 | bool | 是否对KV缓存进行量化 |

**配置示例**:

```yaml
metadata:
  config_id: standing_high
  label:
    w_bit: 8
    a_bit: 8
    is_sparse: false
    kv_cache: false
```

### evaluation - 评估服务配置

**作用**: 定义模型精度评估的配置，包括评估服务类型、测评工具配置和推理引擎配置。

#### type - 评估服务类型

**作用**: 指定评估服务的类型。

**类型**: `string`

**可选值**: `service_oriented`（面向服务的评估，通过服务化方式启动模型进行评估）

#### demand - 精度需求配置

**作用**: 定义模型精度评估的精度需求，包括数据集、目标精度和容差。

**字段说明**:

| 字段名 | 作用 | 类型 | 说明 |
|--------|------|------|------|
| expectations | 精度期望列表 | list | 精度需求的列表，每个元素包含数据集、目标精度和容差 |

**expectations 字段说明**:

| 字段名 | 作用 | 类型 | 说明 |
|--------|------|------|------|
| dataset | 数据集名称 | string | 需要评估的数据集名称，如 gsm8k、aime25 等 |
| target | 目标精度 | float | 期望达到的精度目标值，单位为百分比，如 83 表示 83% 的精度 |
| tolerance | 容差 | float | 精度允许的误差范围，单位为百分比，如 2 表示允许 ±2% 的误差 |

**配置示例**:

```yaml
# 单个数据集的精度需求
demand:
  expectations:
    - dataset: gsm8k
      target: 83  # 目标精度 83%
      tolerance: 2  # 容差 ±2%

# 多个数据集的精度需求
demand:
  expectations:
    - dataset: gsm8k
      target: 83  # 目标精度 83%
      tolerance: 2  # 容差 ±2%
    - dataset: aime25
      target: 85  # 目标精度 85%
      tolerance: 1  # 容差 ±1%
    - dataset: bfcl-simple
      target: 80  # 目标精度 80%
      tolerance: 2  # 容差 ±2%
```

**注意**: 
- 精度目标设置说明：文档中给出的精度数据仅供参考，请根据实际浮点模型的精度进行配置。理论上量化后模型不会超过原始浮点模型的精度，因此建议将精度目标设置为略低于或等于浮点模型的精度。
- 支持配置多个数据集的精度需求，每个数据集可以设置不同的目标精度和容差。

#### evaluation - 测评工具配置

**作用**: 定义测评工具的配置参数。

**核心字段**:

- **type**: 测评工具类型，当前支持 `aisbench`
- **precheck**: 预检查配置（可选），定义正式评估前的预检查配置
- **aisbench**: AISbench 测评工具的详细配置参数
- **datasets**: 数据集配置，定义需要评估的数据集及其配置
- **host**: 服务主机地址
- **port**: 服务端口
- **served_model_name**: 服务化模型名称

**配置示例**:

```yaml
evaluation:
  type: aisbench
  aisbench:
    binary: ais_bench
    mode: all
    timeout: 7200
    request_rate: 1.0
    retry: 2
    batch_size: 32
    max_out_len: 512
    trust_remote_code: false
    pred_postprocessor: extract_non_reasoning_content
    generation_kwargs:
      temperature: 0.5
      top_k: 10
      top_p: 0.9
      seed: null
      repetition_penalty: 1.03
    model_meta:
      base_name: vllm_api_general_chat
      subdir: vllm_api
      abbr: vllm-api-general-chat
      attr: service
    default_metric_keys:
      - final_accuracy
      - accuracy
      - score
  datasets:
    gsm8k:
      config_name: "gsm8k_gen_0_shot_cot_str"
      mode: all
    aime25:
      config_name: "aime2025_gen_0_shot_chat_prompt"
      mode: all
    bfcl-simple:
      config_name: "BFCL_gen_simple"
      mode: all
  host: localhost
  port: 1234
  served_model_name: served_model_name
```

**datasets 字段说明**:

该字段指定了不同的数据集字段对应的 AISbench 拉起测评服务的字段。当前示例中仅支持三个数据集（gsm8k、aime25、bfcl-simple），用户可以参考 [AISbench 文档数据集支持列表](https://gitee.com/aisbench/benchmark)添加更多支持的数据集。每个数据集需要配置 `config_name`（AISbench 中的配置名称）和 `mode`（评估模式）字段。

**aisbench 字段说明**: AISbench 测评工具的详细参数配置请参考 [AISbench 官方文档](https://gitee.com/aisbench/benchmark)。

##### precheck - 预检查配置（可选）

**作用**: 定义正式评估前的预检查配置，用于在每次迭代的模型评估前对量化后的模型进行预验证。

**类型**: `list`

**说明**: precheck 是一个列表，每个元素是一个预检查规则配置，包含 `type` 字段用于指定预检查类型。如果配置了 precheck 且不为空列表，系统会在正式评估前执行预检查。

**支持的预检查类型**:

1. **expected_answer** - 期望答案验证

**expected_answer - 期望答案验证**

**作用**: 验证模型输出是否包含预期的答案内容。

**字段说明**:

| 字段名 | 作用 | 类型 | 说明 |
|--------|------|------|------|
| type | 预检查类型 | string | 固定值：`expected_answer` |
| test_cases | 测试用例列表 | list | 测试用例列表，使用键值对格式（问题: 答案）。如果不配置，默认使用一个测试用例（"What is 2+2?": "4"） |
| max_tokens | 最大token数 | int | 可选，默认值：512 |
| timeout | 超时时间 | float | 可选，默认值：60.0（秒） |

**test_cases 字段说明**:

`test_cases` 使用字典格式（键值对），键为问题，值为答案：

```yaml
test_cases:
  - "What is 2+2?": ["4", "four"]  # 期望响应中包含 "4" 或 "four"
  - "What is the capital of China?": "Beijing"  # 期望响应中包含 "Beijing"
```

**格式说明**：
- 键（问题）：字符串类型，测试消息
- 值（答案）：可以是字符串、字符串列表
  - 字符串：如 `"4"`，表示期望响应中包含 "4"
  - 字符串列表：如 `["4", "four"]`，表示期望响应中包含 "4" 或 "four" 中的任意一个

**配置示例**:

```yaml
precheck:
- type: expected_answer
  test_cases:
    - "What is 2+2?": ["4", "four"]
    - "What is the capital of China?": "Beijing"
  max_tokens: 1024
  timeout: 60.0
```

**注意**:
- 预检查功能会在每次迭代的模型评估前执行（在服务化启动后）。系统会依次执行所有配置的预检查规则，如果任何一个预检查失败，系统会跳过本次迭代的正式评估，返回数据集全零结果，直接开启下一次迭代。
- 预检查的目的是快速发现明显的问题，避免浪费时间进行完整的评估。如果所有预检查都通过，系统会继续进行正式的精度评估。
- **仅支持英文问答**：预检查功能目前仅支持英文问答，测试消息和期望答案应使用英文。
- 如果不配置 `precheck` 字段或配置为空列表，将跳过预检查直接进行正式评估。

#### inference_engine - 推理引擎配置

**作用**: 定义推理引擎的配置参数，用于将量化后的模型以服务化方式启动。

**核心字段**:

- **type**: 推理引擎类型，当前支持 `vllm-ascend`
- **entrypoint**: 服务入口点
- **env_vars**: 环境变量配置
- **served_model_name**: 服务化模型名称
- **host**: 服务主机地址
- **port**: 服务端口
- **health_check_endpoint**: 健康检查端点，用于检查 vLLM-Ascend 是否能正常拉起模型，固定为 `/v1/models`即可，
- **startup_timeout**: 启动超时时间
- **args**: 推理引擎启动参数

**详细参数说明**: vLLM-Ascend 推理引擎的详细参数配置请参考 [vLLM-Ascend 配置指南](https://docs.vllm.ai/projects/vllm-ascend-cn/zh-cn/latest/user_guide/configuration/index.html)。

**注意**: 不同模型拉起服务化时需要的参数可能不同，用户需要根据实际模型调整服务化参数。例如，不同模型可能需要不同的 `max-model-len`、`tensor-parallel-size` 等参数。

**配置示例**:

```yaml
inference_engine:
  type: vllm-ascend
  entrypoint: vllm.entrypoints.openai.api_server
  env_vars:
    HCCL_BUFFSIZE: 1024
    VLLM_VERSION: 0.11.0
    ASCEND_RT_VISIBLE_DEVICES: 0
  served_model_name: served_model_name
  host: localhost
  port: 1234
  health_check_endpoint: /v1/models
  startup_timeout: 600
  args:
    enforce-eager: true
    served-model-name: served_model_name
    trust-remote-code: true
    tensor-parallel-size: 1
    data-parallel-size: 1
    quantization: ascend
    enable-prefix-caching: false
    max-model-len: 8192
    max-num-batched-tokens: 8192
    gpu-memory-utilization: 0.9
    additional_config:
      ascend_scheduler_config:
        enable: true
      enable_weight_nz_layout: true
```

## 配置示例

完整的自动调优配置示例请参考：

- standing_high 调优策略配置：[standing_high.yaml](./example/standing_high.yaml)
