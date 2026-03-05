# 一键量化生成结果

## 简介

本文档详细介绍一键量化功能生成的输出文件结构，帮助用户理解量化结果的组织方式以及各个文件的作用。

## 生成文件概述

执行一键量化命令后，会在指定的 `save_path` 目录下生成以下文件：

```bash
├── config.json                          # 原始模型配置文件
├── generation_config.json               # 原始生成配置文件
├── quant_model_description.json         # 量化权重描述文件
├── quant_model_weights.safetensors  # 量化权重文件（若权重较大可能分片，通过 index.json 索引）
├── tokenizer_config.json                # 原始分词器配置文件
├── tokenizer.json                        # 原始分词器词汇表
├── {model_type}_best_practice.yaml    # 量化配置协议
└── vocab.json                            # 原始词汇映射文件（部分模型）
```

### 文件说明

| 文件名 | 说明 |
|--------|------|
| `config.json` | 原始模型的配置文件，包含模型架构、层数、隐藏维度等关键参数 |
| `generation_config.json` | 原始模型的生成配置文件，包含采样策略、最大生成长度等推理相关参数 |
| `quant_model_description.json` | **量化权重描述文件**，记录每个权重张量的量化类型和元数据 |
| `quant_model_weight_w8a8.safetensors` | **量化权重文件**，包含实际存储的量化后的模型权重数据（若权重较大可能分片保存为多个文件，通过 `model.safetensors.index.json` 索引） |
| `tokenizer_config.json` | 原始分词器的配置文件，包含特殊 token、词表大小等信息 |
| `tokenizer.json` | 原始分词器的词汇表文件，定义 token 与 ID 的映射关系 |
| `{model_type}_best_practice.yaml` | **量化配置协议文件**，记录本次量化所使用的完整配置信息，参考[量化配置协议详解](usage.md#量化配置协议详解) |
| `vocab.json` | 原始词汇映射文件，部分模型（如 GPT 风格模型）会包含此文件 |

> **注意**：文件名中的 `w8a8` 表示量化类型（权重8bit，激活8bit），实际名称取决于用户选择的量化类型，如 `w4a8`、`w8a16` 等。

## quant_model_description.json 详解

`quant_model_description.json` 是量化权重描述文件，它记录了模型中每个权重张量的量化类型和相关元数据信息，是推理框架加载量化模型的重要依据。

### 文件结构示例

```json
{
  "model_quant_type": "W8A8",
  "version": "1.0.0",
  "group_size": 128,
  "kv_quant_type": "KV8",
  "model.layers.0.self_attn.qkv_proj.weight": "W8A8",
  "model.layers.0.self_attn.o_proj.weight": "W8A8",
  "model.layers.0.mlp.gate_proj.weight": "W8A8",
  "model.layers.0.mlp.up_proj.weight": "W8A8",
  "model.layers.0.mlp.down_proj.weight": "W8A8"
}
```

### 字段类型说明

#### 1. 全局元数据字段

| 字段名 | 类型 | 说明 |
|--------|------|------|
| `model_quant_type` | `string` | 模型整体的量化类型，标识模型采用的量化方案 |
| `version` | `string` | 量化工具版本号，格式为 `x.x.x` |
| `group_size` | `int` | 量化分组大小，用于分组量化时的参数 |
| `kv_quant_type` | `string` | KV Cache 量化类型，如 `KV8` 表示 8bit KV Cache |
| `kv_cache_type` | `string` | KV Cache 量化类型的别名，与 `kv_quant_type` 含义相同 |
| `fa_quant_type` | `string` | Flash Attention 量化类型，如 `FAQuant` |
| `reduce_quant_type` | `string` | 通信量化类型，如 `per_channel` |
| `metadata` | `object` | 其他元数据信息，如 QuaRot 相关信息 |
| `optional` | `object` | 可选功能模块信息，如 QuaRot 全局旋转矩阵路径 |

#### 2. 量化类型枚举值

| 枚举值 | 说明 |
|--------|------|
| `FLOAT` | 浮点数（未量化） |
| `W16A16S` | W16A16s稀疏量化，权重16bit稀疏 |
| `W8A8` | W8A8量化，权重8bit，激活8bit |
| `W8A8_DYNAMIC` | W8A8动态量化，权重8bit，激活per-token动态量化 |
| `W8A8_MIX` | W8A8混合量化，结合静态和动态量化 |
| `W8A16` | W8A16量化，权重8bit，激活16bit |
| `W4A4_DYNAMIC` | W4A4动态量化，权重4bit，激活per-token动态量化 |
| `WFP8AFP8_DYNAMIC` | WFP8/AFP8动态量化 |
| `W8A8_MXFP8` | W8A8 MXFP8量化 |
| `W4A8_MXFP` | W4A8 MXFP量化 |
| `W4A4_MXFP4` | W4A4 MXFP4量化 |
| `W4A8_DYNAMIC` | W4A8动态量化，权重4bit，激活per-token动态量化 |
| `C8` | KV Cache 8bit量化 |
| `FAQuant` | Flash Attention量化 |
| `FLATQUANT_DYNAMIC` | FlatQuant动态量化 |
| `FLATQUANT` | FlatQuant静态量化 |

#### 3. 权重张量量化类型

除了上述全局元数据字段外，JSON 文件中的其他键值对表示模型中各个权重张量的量化类型。键为权重名称，值为该权重采用的量化类型。

例如：
```json
{
  "model.layers.0.self_attn.qkv_proj.weight": "W8A8"
}
```

表示 `model.layers.0.self_attn.qkv_proj.weight` 这个权重使用了 W8A8 量化方式。

## 权重类型详解

基于 `quant_model_description.json` 中的字段，本节详细介绍不同量化模式的参数结构。**不同量化模式的量化参数不同，safetensors权重文件和json描述文件也不同。**

### 各量化模式参数详解

以下内容基于 `AscendV1Saver` 实现，详细介绍各量化模式的参数结构。

#### 1. FLOAT（未量化）

FLOAT 表示未进行量化处理的权重，保持原始浮点精度。

**量化参数**（在 safetensors 中的存储）：

| 参数名 | 数据类型 | 说明 |
|--------|----------|------|
| `weight` | float16/bfloat16 | 原始浮点权重 |
| `bias` | float16/bfloat16 | 偏置（可选） |

**典型权重名称**：
```
model.layers.0.self_attn.q_proj.weight
model.layers.0.self_attn.q_proj.bias
```

**quant_model_description.json 标识**：`FLOAT`

---

#### 2. W16A16S（稀疏量化）

W16A16S 是权重浮点稀疏量化模式，权重保持浮点精度并进行稀疏处理。

**量化参数**（在 safetensors 中的存储）：

| 参数名 | 数据类型 | 说明 |
|--------|----------|------|
| `weight` | float16/bfloat16 | 稀疏处理后的权重（非零值） |
| `scale` | float16/bfloat16 | 缩放因子 |

**典型权重名称**：
```
model.layers.0.mlp.gate_proj.weight
model.layers.0.mlp.gate_proj.scale
```

**quant_model_description.json 标识**：`W16A16S`

---

#### 3. W8A8（静态量化）

W8A8 是静态量化模式，对权重和激活值都进行 int8 量化。

**量化参数**（在 safetensors 中的存储）：

| 参数名 | 数据类型 | 说明 |
|--------|----------|------|
| `weight` | int8 | 量化后的权重数据 |
| `quant_bias` | int32 | 量化偏置 |
| `input_scale` | float16/bfloat16 | 激活值量化缩放因子 |
| `input_offset` | float16/bfloat16 | 激活值量化偏移因子 |
| `deq_scale` | int64/float32 | 综合反量化缩放因子 (input_scale × weight_scale) |
| `bias` | float32 | 原始浮点偏置（可选） |

**典型权重名称**：
```
model.layers.0.self_attn.qkv_proj.weight
model.layers.0.self_attn.qkv_proj.quant_bias
model.layers.0.self_attn.qkv_proj.input_scale
model.layers.0.self_attn.qkv_proj.input_offset
model.layers.0.self_attn.qkv_proj.deq_scale
```

**quant_model_description.json 标识**：`W8A8`

---

#### 4. W8A8_DYNAMIC（动态量化）

W8A8_DYNAMIC 是权重 int8 量化、激活值 per-token 动态量化的模式。

**量化参数**（在 safetensors 中的存储）：

| 参数名 | 数据类型 | 说明 |
|--------|----------|------|
| `weight` | int8 | 量化后的权重数据 |
| `weight_scale` | float32 | 权重量化缩放因子 |
| `weight_offset` | float32 | 权重量化偏移因子（对称量化时为全0） |
| `bias` | float32 | 原始浮点偏置（可选） |

**典型权重名称**：
```
model.layers.0.self_attn.qkv_proj.weight
model.layers.0.self_attn.qkv_proj.weight_scale
model.layers.0.self_attn.qkv_proj.weight_offset
```

**反量化公式**：
```python
deq_weight = (weight - weight_offset) * weight_scale
```

**quant_model_description.json 标识**：`W8A8_DYNAMIC`

---

#### 5. W8A8_MIX（混合量化）

W8A8_MIX 是结合静态和动态量化的混合模式。

**量化参数**（在 safetensors 中的存储）：

| 参数名 | 数据类型 | 说明 |
|--------|----------|------|
| `weight` | int8 | 量化后的权重数据 |
| `quant_bias` | int32 | 量化偏置 |
| `input_scale` | float16/bfloat16 | 激活值量化缩放因子 |
| `input_offset` | float16/bfloat16 | 激活值量化偏移因子 |
| `deq_scale` | int64/float32 | 综合反量化缩放因子 |
| `weight_scale` | float32 | 权重量化缩放因子 |
| `weight_offset` | float32 | 权重量化偏移因子 |
| `bias` | float32 | 原始浮点偏置（可选） |

**典型权重名称**：
```
model.layers.0.self_attn.qkv_proj.weight
model.layers.0.self_attn.qkv_proj.quant_bias
model.layers.0.self_attn.qkv_proj.input_scale
model.layers.0.self_attn.qkv_proj.input_offset
model.layers.0.self_attn.qkv_proj.deq_scale
model.layers.0.self_attn.qkv_proj.weight_scale
model.layers.0.self_attn.qkv_proj.weight_offset
```

**quant_model_description.json 标识**：`W8A8_MIX`

---

#### 6. W8A16（权重量化）

W8A16 是权重量化模式，只对权重进行 int8 量化，激活值保持浮点精度。

**量化参数**（在 safetensors 中的存储）：

| 参数名 | 数据类型 | 说明 |
|--------|----------|------|
| `weight` | int8 | 量化后的权重数据 |
| `weight_scale` | float32 | 权重量化缩放因子 |
| `weight_offset` | float32 | 权重量化偏移因子（对称量化时为全0） |
| `bias` | float32 | 原始浮点偏置（可选） |

**典型权重名称**：
```
model.layers.0.self_attn.qkv_proj.weight
model.layers.0.self_attn.qkv_proj.weight_scale
model.layers.0.self_attn.qkv_proj.weight_offset
```

**反量化公式**：
```python
deq_weight = (weight - weight_offset) * weight_scale
```

**quant_model_description.json 标识**：`W8A16`

---

#### 7. W4A4_DYNAMIC（W4A4 动态量化）

W4A4_DYNAMIC 是权重 int4 量化、激活值 per-token 动态量化的极低比特模式。

**量化参数**（在 safetensors 中的存储）：

| 参数名 | 数据类型 | 说明 |
|--------|----------|------|
| `weight` | int8 | 量化后的权重数据（int4 打包存储） |
| `weight_scale` | float32 | 权重量化缩放因子 |
| `weight_offset` | float32 | 权重量化偏移因子 |
| `bias` | float32 | 原始浮点偏置（可选） |

**典型权重名称**：
```
model.layers.0.self_attn.qkv_proj.weight
model.layers.0.self_attn.qkv_proj.weight_scale
model.layers.0.self_attn.qkv_proj.weight_offset
```

**重要说明**：激活值量化参数在推理时动态计算，不保存到权重文件中。

**quant_model_description.json 标识**：`W4A4_DYNAMIC`

---

#### 8. WFP8AFP8_DYNAMIC（FP8 动态量化）

WFP8AFP8_DYNAMIC 是 FP8 浮点动态量化模式。

**量化参数**（在 safetensors 中的存储）：

| 参数名 | 数据类型 | 说明 |
|--------|----------|------|
| `weight` | float8_e4m3fn | FP8 格式权重 |
| `weight_scale` | float32 | 权重量化缩放因子 |
| `weight_offset` | float32 | 权重量化偏移因子 |
| `bias` | float32 | 原始浮点偏置（可选） |

**典型权重名称**：
```
model.layers.0.self_attn.qkv_proj.weight
model.layers.0.self_attn.qkv_proj.weight_scale
model.layers.0.self_attn.qkv_proj.weight_offset
```

**quant_model_description.json 标识**：`WFP8AFP8_DYNAMIC`

---

#### 9. W8A8_MXFP8（MXFP8 量化）

W8A8_MXFP8 是使用 MX（Mixtral 格式）FP8 的量化模式。

**量化参数**（在 safetensors 中的存储）：

| 参数名 | 数据类型 | 说明 |
|--------|----------|------|
| `weight` | float8_e4m3fn | FP8 格式权重 |
| `weight_scale` | uint8 | 缩放因子（+127 偏移后存储） |
| `bias` | float32 | 原始浮点偏置（可选） |

**说明**：`weight_scale` 进行了 +127 偏移处理，使其从 -127~128 偏移到 0~255，正好覆盖 `uint8` 的取值范围。

**典型权重名称**：
```
model.layers.0.self_attn.qkv_proj.weight
model.layers.0.self_attn.qkv_proj.weight_scale
```

**quant_model_description.json 标识**：`W8A8_MXFP8`

---

#### 10. W4A8_MXFP（W4A8 MXFP 量化）

W4A8_MXFP 是权重 int4 + 激活 int8 的 MXFP 量化模式。

**量化参数**（在 safetensors 中的存储）：

| 参数名 | 数据类型 | 说明 |
|--------|----------|------|
| `weight` | float8_e4m3fn | FP8 格式权重 |
| `weight_scale` | uint8 | 缩放因子（+127 偏移后存储） |
| `bias` | float32 | 原始浮点偏置（可选） |

**典型权重名称**：
```
model.layers.0.self_attn.qkv_proj.weight
model.layers.0.self_attn.qkv_proj.weight_scale
```

**quant_model_description.json 标识**：`W4A8_MXFP`

---

#### 11. W4A4_MXFP4（W4A4 MXFP4 量化）

W4A4_MXFP4 是极低比特的 MXFP4 量化模式。

**量化参数**（在 safetensors 中的存储）：

| 参数名 | 数据类型 | 说明 |
|--------|----------|------|
| `weight` | float8_e4m3fn | FP8 格式权重 |
| `weight_scale` | uint8 | 缩放因子（+127 偏移后存储） |
| `bias` | float32 | 原始浮点偏置（可选） |

**典型权重名称**：
```
model.layers.0.self_attn.qkv_proj.weight
model.layers.0.self_attn.qkv_proj.weight_scale
```

**quant_model_description.json 标识**：`W4A4_MXFP4`

---

#### 12. C8（KV Cache 量化）

KV Cache 8bit 量化是针对 Key-Value 缓存的量化模式。

**量化参数**（在 safetensors 中的存储）：

| 参数名 | 数据类型 | 说明 |
|--------|----------|------|
| `kv_cache_scale` | float32/float16 | KV Cache 量化缩放因子 |
| `kv_cache_offset` | float32/float16 | KV Cache 量化偏移因子 |

**典型权重名称**：
```
model.layers.0.self_attn.k_proj.kv_cache_scale
model.layers.0.self_attn.k_proj.kv_cache_offset
model.layers.0.self_attn.v_proj.kv_cache_scale
model.layers.0.self_attn.v_proj.kv_cache_offset
```

**quant_model_description.json 标识**：`C8`

---

#### 13. W4A8_DYNAMIC（W4A8 动态量化）

W4A8_DYNAMIC 是权重 int4 量化、激活值 per-token 动态量化的模式。

**量化参数**（在 safetensors 中的存储）：

| 参数名 | 数据类型 | 说明 |
|--------|----------|------|
| `weight` | int8 | 量化后的权重数据（int4 打包存储） |
| `weight_scale` | float32 | 权重量化缩放因子 |
| `weight_offset` | float32 | 权重量化偏移因子 |
| `scale_bias` | float32 | 缩放偏置因子（用于反量化时的额外调整） |
| `bias` | float32 | 原始浮点偏置（可选） |

**典型权重名称**：
```
model.layers.0.self_attn.qkv_proj.weight
model.layers.0.self_attn.qkv_proj.weight_scale
model.layers.0.self_attn.qkv_proj.weight_offset
model.layers.0.self_attn.qkv_proj.scale_bias
```

**反量化公式**：
```python
deq_weight = (weight - weight_offset) * weight_scale + scale_bias
```

**quant_model_description.json 标识**：`W4A8_DYNAMIC`

---

#### 14. FlatQuant_DYNAMIC（FlatQuant 动态量化）

FlatQuant_DYNAMIC 是 FlatQuant 动态量化模式。

**量化参数**（在 safetensors 中的存储）：

| 参数名 | 数据类型 | 说明 |
|--------|----------|------|
| `weight` | int8/int32 | 量化后的权重数据 |
| `weight_scale` | float32 | 权重量化缩放因子 |
| `weight_offset` | float32 | 权重量化偏移因子 |
| `input_scale` | float32 | 激活值量化缩放因子（动态） |
| `input_offset` | float32 | 激活值量化偏移因子（动态） |
| `deq_scale` | float32 | 综合反量化缩放因子 |
| `quant_bias` | int32 | 量化偏置 |
| `left_trans` | float32 | 线性变换左矩阵 |
| `right_trans` | float32 | 线性变换右矩阵 |
| `clip_ratio` | float32 | 裁剪比例因子 |
| `bias` | float32 | 原始浮点偏置（可选） |

**说明**：
- FlatQuant 是一种结合线性变换的量化方法
- `left_trans` 和 `right_trans` 是用于特征变换的矩阵
- `clip_ratio` 用于控制量化范围

**quant_model_description.json 标识**：`W8A8_FLATQUANT_DYNAMIC` 或 `W4A8_FLATQUANT_DYNAMIC`

---

#### 15. NonFusionSmoothQuant（平滑量化）

NonFusionSmoothQuant 是一种平滑量化模式，用于减少量化误差。

**量化参数**（在 safetensors 中的存储）：

| 参数名 | 数据类型 | 说明 |
|--------|----------|------|
| `div.mul_scale` | float32 | 平滑缩放因子 |
| 其他参数 | - | 由内部 Linear 层的量化类型决定 |

**典型权重名称**：
```
model.layers.0.self_attn.q_proj.div.mul_scale
model.layers.0.self_attn.q_proj.linear.weight
```

**quant_model_description.json 标识**：`FLOAT`（内层权重）

---

#### 16. QuaRot（旋转量化）

QuaRot 是一种基于旋转的量化方法，用于保持量化后模型的功能等价性。

**量化参数**（在 safetensors 中的存储）：

| 参数名 | 数据类型 | 说明 |
|--------|----------|------|
| `heads_rotation` | float32 | 多头注意力旋转矩阵 |
| `kronecker_rotation_m` | float32 | Kronecker 旋转矩阵 M |
| `kronecker_rotation_n` | float32 | Kronecker 旋转矩阵 N |
| `global_rotation` | float32 | 全局旋转矩阵（保存在 optional 目录） |

**说明**：
- `heads_rotation` 用于多头注意力的旋转
- `kronecker_rotation_m` 和 `kronecker_rotation_n` 用于 MLP 层的旋转
- `global_rotation` 保存在 `optional/quarot.safetensors` 文件中

**quant_model_description.json 标识**：`optional.quarot.global_rotation`（路径指向 optional 目录） 

---

#### 17. FAQuant（Flash Attention 量化）

FAQuant 是 Flash Attention 量化模式。

**量化参数**（在 safetensors 中的存储）：

| 参数名 | 数据类型 | 说明 |
|--------|----------|------|
| `scale` | float16/bfloat16 | 量化缩放因子 |
| `offset` | float16/bfloat16 | 量化偏移因子 |

**典型权重名称**：
```
model.layers.0.self_attn.q_proj.scale
model.layers.0.self_attn.q_proj.offset
```

**quant_model_description.json 标识**：`FAQuant`

> 注：
> 1. ✓ 表示该量化模式包含此参数，- 表示不包含，✓ (+127) 表示需要 +127 偏移处理
> 2. NonFusionSmoothQuant 的参数由内部 Linear 层的量化类型决定，额外包含 `div.mul_scale` 参数
> 3. QuaRot 的旋转矩阵参数根据具体实现可能包含部分或全部旋转矩阵
