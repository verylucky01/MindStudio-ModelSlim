# AWQ：激活感知权重量化算法说明

## 简介

- **概述**：AWQ（Activation-aware Weight Quantization，激活感知权重量化）是一种用于大语言模型量化过程中抑制激活离群值的算法。该算法通过观察激活值的统计特征，自动搜索最优的缩放因子，在权重量化前对其进行缩放，从而在保持模型精度的同时有效减少量化误差。AWQ的核心理念是：并非所有权重对模型输出同等重要，通过激活值分布来识别重要权重通道并给予保护，可以在低比特量化场景下获得更优的精度表现。
- **核心思想**：AWQ算法的核心思想是利用激活值的均值（mean）来度量每个权重通道的重要性，通过网格搜索（grid search）找到使量化误差最小的缩放因子。算法对每个子图中的线性层权重进行缩放后量化，通过对比量化结果与浮点基准结果的MSE误差，选择最优的缩放比例。

## 使用前准备

安装 msModelSlim 工具，详情请参见[《msModelSlim工具安装指南》](../../getting_started/install_guide.md)。

## 原理和实现

### 原理

AWQ算法使用以下公式计算缩放因子：

```
scales = act_mean.pow(ratio).clamp(min=1e-4)
scales = scales / sqrt(scales.max() * scales.min())
```

其中：

- `act_mean`：激活值绝对值的逐通道均值（`mean(abs(act))`），反映各通道的重要性。
- `ratio`：缩放比例系数，在 `[0, 1)` 范围内以 `1/n_grid` 为步长进行网格搜索。
- `n_grid`：网格搜索的步数，默认值为20，控制搜索精度。

**关键特性：**

1. **基于激活均值的重要性评估**：使用激活值绝对值的逐通道均值来反映权重通道的重要性，均值越大的通道在量化时受到更多保护。
2. **网格搜索最优缩放**：在 `[0, 1)` 范围内以 `1/n_grid` 为步长遍历不同的 `ratio` 值，评估每个候选缩放因子下的量化误差。
3. **实际量化器评估**：使用真实的权重量化器（`AutoWeightQuantizer`）对缩放后的权重进行量化，计算量化结果与浮点基准的MSE误差。
4. **块级（block-level）误差评估**：通过自动发现目标模块的最低公共祖先（LCA）并缓存其输入参数，在整个块级别评估量化误差，而非仅关注单个线性层的权重误差。

**缩放因子搜索流程**

1. **初始化**：收集激活值均值和祖先模块（通过LCA自动发现）的输入参数缓存。
2. **基准推理**：使用原始浮点权重在祖先模块上进行推理，得到浮点基准输出（golden outputs）。
3. **网格搜索**：在 `[0, 1)` 范围内以 `1/n_grid` 为步长遍历 `ratio` 值：
   - 计算缩放因子：`scales = act_mean.pow(ratio).clamp(min=1e-4)`
   - 归一化缩放因子：`scales = scales / sqrt(max(scales) * min(scales))`
   - 对目标线性层的权重应用缩放：`weight *= scales`
   - 使用量化器量化缩放后的权重
   - 反向缩放量化后的权重：`weight = quantized_weight / scales`
   - 在中间模块上推理并计算MSE误差
   - 恢复原始权重
4. **选择最优参数**：选择使MSE误差最小的 `ratio` 对应的缩放因子。
5. **应用缩放**：通过 `SubgraphFusionFactory` 将最优缩放因子融合到子图的权重中。

### 支持的子图类型

AWQ算法支持五种子图类型：

#### 1. NonFusionSubgraph（非融合子图）

适用于不需要融合操作的场景，即 `MappingConfig` 中 `source` 为 `None`，仅指定 `targets` 线性层列表。当模型结构不符合标准的 norm-linear / linear-linear / ov / up-down 模式时，可通过非融合子图对目标线性层直接进行AWQ缩放。

**处理方式：**

- 使用所有 `targets` 线性层进行缩放因子搜索。
- 通过自动发现的最低公共祖先（LCA）模块进行块级误差评估。
- 搜索到最优缩放因子后，仅对目标线性层应用正向缩放（不涉及源模块的反向缩放）。

**触发条件：** 在 `AdapterConfig` 中将 `MappingConfig.source` 设为 `None`，框架会自动将其作为非融合子图处理。

#### 2. NormLinearSubgraph（归一化-线性子图）

适用于包含归一化层和多个线性层的结构，如：

```python
x = norm(x)
y = torch.cat([linear(x) for linear in linears], dim=-1)
```

**处理方式：**

- 使用所有目标线性层进行缩放因子搜索。
- 通过自动发现的最低公共祖先（LCA）模块进行块级误差评估。
- 搜索到最优缩放因子后，对归一化层应用反向缩放（`1/scales`），对线性层应用正向缩放。

#### 3. LinearLinearSubgraph（线性-线性子图）

适用于两个连续线性层的结构：

```python
y = linear2(linear1(x))
```

**处理方式：**

- 基于 `linear2` 的权重进行缩放因子搜索。
- 通过自动发现的LCA模块进行块级误差评估。
- 对 `linear2` 应用正向缩放，对 `linear1` 应用反向缩放（`1/scales`）。

#### 4. OVSubgraph（注意力输出-值子图）

适用于注意力机制中的输出投影和值投影：

- 支持MHA（多头注意力）
- 支持MQA（多查询注意力）
- 支持GQA（分组查询注意力）

**处理方式：**

- 基于 `o_proj` 权重进行缩放因子搜索。
- 通过自动发现的LCA模块进行块级误差评估。
- 对 `o_proj` 应用正向缩放，对 `v_proj` 应用反向缩放（`1/scales`）。

#### 5. UpDownSubgraph（上投影-下投影子图）

适用于MLP门控机制：

```python
y = down_proj(ReLU(gate_proj(x)) * up_proj(x))
```

**处理方式：**

- 基于 `down_proj` 权重进行缩放因子搜索。
- 通过自动发现的LCA模块进行块级误差评估。
- 对 `down_proj` 应用正向缩放，对 `up_proj` 应用反向缩放（`1/scales`）。

### 代码架构

AWQ算法的代码组织在 `msmodelslim/processor/anti_outlier/awq/` 目录下：

| 文件                                                                                                  | 核心类/函数                                        | 职责                                                 |
| ----------------------------------------------------------------------------------------------------- | -------------------------------------------------- | ---------------------------------------------------- |
| [`processor.py`](../../../../msmodelslim/processor/anti_outlier/awq/processor.py)                     | `AWQProcessor`, `AWQProcessorConfig`               | Processor入口，管理预处理/后处理流程，自动发现LCA    |
| [`api.py`](../../../../msmodelslim/processor/anti_outlier/awq/api.py)                                 | `awq()`, `awq_impl_*()`                            | 各子图类型的AWQ分发与实现                            |
| [`best_scales_search.py`](../../../../msmodelslim/processor/anti_outlier/awq/best_scales_search.py)   | `AWQSearcher`, `AWQBestScalesSearcher`             | 缩放因子网格搜索逻辑                                 |
| [`awq_stats_collector.py`](../../../../msmodelslim/processor/anti_outlier/awq/awq_stats_collector.py) | `AWQStatsCollector`                                | 激活统计信息收集（均值、kwargs缓存），使用全局上下文 |
| [`common.py`](../../../../msmodelslim/processor/anti_outlier/awq/common.py)                           | `AWQConfig`, `AWQContext`, `offload()`, `onload()` | 算法配置、运行时上下文、张量设备迁移工具             |
| [`interface.py`](../../../../msmodelslim/processor/anti_outlier/awq/interface.py)                     | `AWQInterface`                                     | 模型适配器需要实现的抽象接口                         |

**关键类说明：**

- **`AWQConfig`**：算法配置，包含 `version`（分发版本号）和 `awq_searcher`（`AWQSearcher` 实例）。
- **`AWQContext`**：运行时上下文，包含 `act_mean`（逐通道激活均值张量）、`inspect_module`（用于块级评估的祖先模块）和 `inspect_module_args`（缓存的输入参数列表，通过 `offload()` 卸载到 CPU）。
- **`AWQStatsCollector`**：通过 `HookManager` 为目标模块安装 forward hook（收集激活均值）和 forward pre-hook（缓存 kwargs），统计数据存储在全局 `IContext` 的 AWQ 命名空间中，支持增量均值更新。
- **`offload()` / `onload()`**：递归地将值树中的所有张量迁移到指定设备，用于 CPU 卸载以节省显存。`offload()` 默认移至 CPU，`onload()` 在推理时将张量移回计算设备。

### 实现

处理流程分两阶段：

#### 1) 预处理阶段（preprocess）

**子图发现与构建：**

- 通过 `AWQInterface` 的 `get_adapter_config_for_subgraph()` 方法获取全局子图信息，识别子图类型：`norm-linear`、`linear-linear`、`ov`、`up-down`。当 `MappingConfig.source` 为 `None` 时自动识别为非融合子图（`NonFusionSubgraph`）。
- 根据配置的 `include/exclude` 模式过滤子图。

**统计信息收集：**

- 为子图中的目标线性模块安装前向钩子（forward hook），收集激活值均值统计信息。
- 通过 `_find_lowest_common_ancestor()` 自动发现目标模块的最低公共祖先（LCA），并为其安装前向预钩子（forward pre-hook），缓存输入参数（kwargs），用于后续块级误差评估。无需在 `MappingConfig` 中手动指定 `module2inspect`。
- 统计数据存储在全局 `IContext` 的 AWQ 命名空间（`IValidatedState`）中，而非 Processor 本地字典。
- 钩子在 `[batch, seq, hidden_dim]` 维度上收集激活值统计信息：
  - **激活均值**：收集激活值绝对值的逐通道均值，使用增量均值算法支持多批次数据。
  - **中间参数缓存**：使用普通 `list` 配合 `offload()` 函数缓存中间模块的输入参数，自动将张量卸载到 CPU 以节省显存。

#### 2) 后处理阶段（postprocess）

**按优先级处理子图：**

- 按默认配置的优先级顺序处理：`up-down`（最高）→ `ov`（高）→ `norm-linear`（中）→ `linear-linear`（低）。
- 每种子图类型调用相应的AWQ处理方法。

**子图AWQ处理：**

- **构建AWQ上下文**：从全局上下文中获取目标模块的激活均值，通过LCA自动发现祖先模块并获取其输入参数缓存，通过 `_resolve_module()` 解析祖先模块实例，构建 `AWQContext`。推理时通过 `onload()` 将参数移回计算设备。
- **执行AWQ搜索**：调用 `AWQBestScalesSearcher.search()` 方法，在祖先模块上搜索最优缩放因子。
- **应用缩放因子**：通过 `SubgraphFusionFactory` 将最优缩放因子融合到子图中，调整源模块和目标模块的权重。

**资源清理：**

- 停止所有钩子的观测。
- 清理统计信息内存。

## 适用要求

- **模型架构要求**：模型必须支持 `AWQInterface` 接口，并正确配置子图映射关系。
- **模块命名要求**：模块名称必须与 `named_modules()` 返回的完整路径完全一致。
- **子图类型支持**：目前支持四种标准融合子图类型（`norm-linear`、`linear-linear`、`ov`、`up-down`）及非融合子图（`source` 为 `None` 时自动触发）。
- **模块属性要求**：目标模块必须存在且具备可写的 `weight`，其他自定义模块暂不支持。
- **模型结构假设**：算法基于标准的Transformer架构设计，对于非标准结构需要谨慎评估适用性。
- **运行时上下文要求**：AWQ 依赖 `ContextManager` 提供全局 `IContext`，Runner（`GeneratedRunner` / `LayerWiseRunner`）会在执行时自动创建和管理上下文。

## 功能介绍

### 使用说明

**导入路径：**

```python
from msmodelslim.processor.anti_outlier.awq import AWQProcessor, AWQProcessorConfig, AWQInterface
from msmodelslim.core.graph.adapter_types import AdapterConfig, MappingConfig, FusionConfig
```

作为 Processor 使用

```yaml
- type: "awq" # 固定为 `awq`，用于指定 Processor。
  weight_qconfig: # 权重量化配置，为必填参数。
    scope: "per_channel" # 量化范围：per_channel。
    dtype: "int4" # 量化数据类型：int4 或 int8。
    symmetric: True # 是否对称量化：True 或 False。
    method: "minmax" # 量化方法：minmax。
  n_grid: 20 # 网格搜索步数，默认值为20，值越大搜索越精细但耗时更长。
  enable_subgraph_type: # 字符串列表，指定启用的子图类型，默认启用所有四种类型。
    - "norm-linear"
    - "linear-linear"
    - "ov"
    - "up-down"
  include: # 包含的层，支持通配符。
    - "*"
  exclude: # 排除的层，支持通配符。
    - ""
```

### YAML配置示例

```yaml
spec:
  process:
    - type: "awq"
      weight_qconfig: # 权重量化配置，为必填参数。
        scope: "per_channel"
        dtype: "int4"
        symmetric: True
        method: "minmax"
      n_grid: 20 # 网格搜索步数，默认为20。
      enable_subgraph_type: # 开启的子图类型。
        - "norm-linear"
        - "linear-linear"
        - "ov"
        - "up-down"
      include: ["*"] # 包含的层，支持通配符。
      exclude: [] # 排除的层，支持通配符。
```

### YAML配置字段详解

| 字段名               | 作用           | 说明                                                                                |
| -------------------- | -------------- | ----------------------------------------------------------------------------------- |
| type                 | 处理器类型标识 | 固定值"awq"，用于标识这是一个激活感知权重量化处理器。                               |
| weight_qconfig       | 权重量化配置   | 必填参数，包含scope、dtype、symmetric、method等字段，用于搜索过程中的实际量化评估。 |
| n_grid               | 网格搜索步数   | 正整数，默认值为20，控制搜索精度，值越大搜索越精细但耗时更长。                      |
| enable_subgraph_type | 开启的子图类型 | 支持的子图类型列表，包括"norm-linear"、"linear-linear"、"ov"、"up-down"。           |
| include              | 包含的层       | 支持通配符匹配。                                                                    |
| exclude              | 排除的层       | 支持通配符匹配。                                                                    |

## 模型适配

### 接口与数据结构

AWQ使用以下接口和数据结构：

```python
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from abc import ABC, abstractmethod

@dataclass
class MappingConfig:
    """模块映射关系配置"""
    targets: List[str]  # 目标模块名称列表，如 ["model.layers.0.self_attn.q_proj", "model.layers.0.self_attn.k_proj"]
    source: Optional[str] = None  # 源模块名称，如 "model.layers.0.input_layernorm"，设为None时为非融合子图

@dataclass
class FusionConfig:
    """融合配置，支持QKV融合等高级功能"""
    fusion_type: str = "none"  # 融合类型：none, qkv, custom等
    num_attention_heads: Optional[int] = None  # 注意力头数量
    num_key_value_heads: Optional[int] = None  # 键值头数量
    custom_config: Optional[Dict[str, Any]] = None  # 自定义配置

@dataclass
class AdapterConfig:
    """子图适配器配置"""
    subgraph_type: str  # 子图类型：norm-linear, linear-linear, ov, up-down
    mapping: Optional[MappingConfig] = None  # 模块映射关系
    fusion: FusionConfig = field(default_factory=lambda: FusionConfig())  # 融合配置

# 模型适配AWQ算法接口
class AWQInterface(ABC):
    @abstractmethod
    def get_adapter_config_for_subgraph(self) -> List[AdapterConfig]:
        """
        返回模型中所有可进行AWQ处理的子图配置

        Returns:
            List[AdapterConfig]: 子图配置列表，每个配置包含：
                - subgraph_type: 子图类型
                - mapping: 源模块到目标模块的映射关系
                - fusion: 融合配置（如QKV融合）
        """
        pass
```

### 适配步骤

**前置要求：**

- 模型需要继承 `AWQInterface` 接口。
- 模块名称必须与 `named_modules()` 返回的完整路径一致。
- 支持的子图类型：`norm-linear`、`linear-linear`、`ov`、`up-down`。
- 配置中的 `subgraph_type`、`mapping` 是必要参数。
- 块级误差评估的父模块通过最低公共祖先（LCA）算法自动发现，无需手动配置。
- 当配置 `FusionConfig` 且 `fusion_type` 为 `qkv` 时，必须给出 `num_attention_heads` 和 `num_key_value_heads`。

**步骤：**

1. **继承接口**：模型适配器继承 `AWQInterface` 接口，实现 `get_adapter_config_for_subgraph()` 方法。
2. **配置子图映射**：为每层配置子图映射关系，框架会自动通过LCA算法发现评估用的父模块：
   - **非融合子图**：将 `source` 设为 `None`，仅指定 `targets` 线性层列表，框架自动作为非融合子图处理。
   - **Norm-Linear子图**：归一化层到后续线性层的映射。
   - **OV子图**：V投影到O投影的映射。
   - **Up-Down子图**：上投影到下投影的映射。
   - **Linear-Linear子图**：连续线性层的映射。
3. **指定模块路径**：使用完整的模块路径，如 `model.layers.{i}.self_attn.q_proj`。

**参考实现：** 可参考 [`msmodelslim/model/qwen2/model_adapter.py`](../../../../msmodelslim/model/qwen2/model_adapter.py) 中的 [`Qwen2ModelAdapter`](../../../../msmodelslim/model/qwen2/model_adapter.py#L40) 实现。

### 配置示例

以下是一个典型的Transformer层配置示例：

```python
def get_adapter_config_for_subgraph(self) -> List[AdapterConfig]:
    adapter_configs: List[AdapterConfig] = []
    for i in range(self.config.num_hidden_layers):
        layer_prefix = f"model.layers.{i}"

        # 1. 输入层归一化到QKV投影的Norm-Linear映射
        norm_linear_attn = MappingConfig(
            source=f"{layer_prefix}.input_layernorm",
            targets=[
                f"{layer_prefix}.self_attn.k_proj",
                f"{layer_prefix}.self_attn.q_proj",
                f"{layer_prefix}.self_attn.v_proj",
            ],
        )

        # 2. 后注意力层归一化到MLP投影的Norm-Linear映射
        norm_linear_mlp = MappingConfig(
            source=f"{layer_prefix}.post_attention_layernorm",
            targets=[
                f"{layer_prefix}.mlp.gate_proj",
                f"{layer_prefix}.mlp.up_proj",
            ],
        )

        # 3. MLP门控机制的Up-Down映射
        up_down_mapping = MappingConfig(
            source=f"{layer_prefix}.mlp.up_proj",
            targets=[f"{layer_prefix}.mlp.down_proj"],
        )

        # 4. 注意力机制中的OV映射（source为v_proj，targets为o_proj）
        ov_mapping = MappingConfig(
            source=f"{layer_prefix}.self_attn.v_proj",
            targets=[f"{layer_prefix}.self_attn.o_proj"],
        )

        # 5. 非融合子图示例（source为None，仅指定targets，LCA自动发现评估父模块）
        non_fusion_mapping = MappingConfig(
            source=None,
            targets=[
                f"{layer_prefix}.self_attn.q_proj",
                f"{layer_prefix}.self_attn.k_proj",
            ],
        )

        adapter_configs.extend([
            AdapterConfig(subgraph_type="norm-linear", mapping=norm_linear_attn),
            AdapterConfig(subgraph_type="norm-linear", mapping=norm_linear_mlp),
            AdapterConfig(subgraph_type="ov", mapping=ov_mapping),
            AdapterConfig(subgraph_type="up-down", mapping=up_down_mapping),
            # 非融合子图：subgraph_type仍需为有效类型，source=None触发非融合处理
            AdapterConfig(subgraph_type="norm-linear", mapping=non_fusion_mapping),
        ])

    return adapter_configs
```

### 与 Flex AWQ SSZ 的区别

| 特性             | AWQ                                       | Flex AWQ SSZ                            |
| ---------------- | ----------------------------------------- | --------------------------------------- |
| 缩放因子计算     | `act_mean.pow(ratio)` 网格搜索            | `A_scale**alpha / W_scale**beta`        |
| 误差评估方式     | 块级评估（通过LCA自动发现的祖先模块推理） | 使用 `LinearQuantizer` 评估             |
| 参数搜索空间     | `ratio` ∈ `[0, 1)`，步长 `1/n_grid`       | `alpha` ∈ `[0, 1]`，步长 0.05，`beta=0` |
| 配置接口         | `AWQInterface`                            | `FlexSmoothQuantInterface`              |
| `module2inspect` | 自动发现（LCA算法），无需手动配置         | 不需要                                  |
| `qconfig`        | 仅需权重量化配置                          | 需要激活+权重量化配置                   |

## FAQ

### 1. 模块名不匹配

**现象**: `include/exclude` 未命中时，日志提示未匹配模式。  
**解决方案**: 核对完整模块名是否与 `named_modules()` 返回的路径一致。

### 2. 子图配置错误

**现象**: `get_adapter_config_for_subgraph()` 返回的配置不正确。  
**解决方案**: 检查配置中的 `source` 和 `targets` 字段是否正确。

### 3. 模块不存在

**现象**: 配置中指定的模块名称在模型中不存在。  
**解决方案**: 通过 `model.named_modules()` 验证模块是否确实存在。

### 4. 子图类型不支持

**现象**: 配置的子图类型不被支持。  
**解决方案**: 确保配置的子图类型在 `enable_subgraph_type` 列表中。

### 5. 祖先模块未找到

**现象**: 日志提示 "No name found for inspect module of subgraph"，子图被跳过。  
**解决方案**: 检查 `targets` 中的模块名称是否具有合理的共同路径前缀，确保其最低公共祖先模块在模型中存在。

### 6. 激活统计信息缺失

**现象**: 日志提示 "No activation mean for target module"，子图被跳过。  
**解决方案**: 确保校准数据（calibration data）足够且模型前向推理正常执行，使钩子能够正确收集激活统计信息。

### 7. 中间参数缓存为空

**现象**: 日志提示 "No kwargs cache for parent module"，子图被跳过。  
**解决方案**: 确保通过LCA自动发现的祖先模块在前向推理中被正确触发，检查 `targets` 中的模块路径是否正确。
