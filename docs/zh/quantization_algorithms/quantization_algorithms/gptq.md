# GPTQ：权重量化算法说明

## 简介

- **问题**：传统量化方法（如MinMax）在权重分布不均匀时，量化误差较大，影响模型精度。
- **目标**：通过逐列优化方式，将量化误差在后续未权重中进行补偿，进而达到最小化量化误差，提升量化模型的精度。

## 使用前准备

安装 msModelSlim 工具，详情请参见[《msModelSlim工具安装指南》](../../getting_started/install_guide.md)。

## 原理和实现

### 原理

通过逐层逐列对权重进行量化，根据量化误差和激活值的hessian矩阵对未量化的权重进行补偿，以此达到整体量化权重误差最小的目的。

**核心思想**

1. **逐层量化**：对模型每一层独立量化，避免误差积累。
2. **基于二阶信息的误差修正**：利用Hessian矩阵评估权重量化对输出的影响，并动态调整未量化权重以补偿误差。
3. **分块量化**：将权重划分成多个块，减少计算复杂度。
4. **惰性批量更新**：在更新未量化权重时，延迟更新，将多个更新合并执行，减少计算开销。

### 实现

算法实现在 [`msmodelslim/core/quantizer/impl/gptq.py`](https://gitcode.com/Ascend/msmodelslim/blob/master/msmodelslim/core/quantizer/impl/gptq.py) 中：
  - `per_channel`实现类：`WeightPerChannelGPTQ`
  - `per_group`实现类：`WeightPerGroupGPTQ`

## 适用要求

- **高精度需求**：适用于对精度要求较高的模型量化场景。
- **计算成本**：GPTQ算法需要计算激活值的hessian矩阵和矩阵分解，且对权重逐列量化，量化速度较慢，计算成本较大。
- **使用限制**：
    - 目前支持int8场景的 per_channel/per_group 对称与非对称量化。
    - 当前暂不支持int4场景的 per_channel/per_group 对称与非对称量化（后续将支持）。
    - 由于GPTQ量化算法依赖模型激活值，MOE模型量化场景要求校准集覆盖所有专家，对校准集要求较高，此场景不推荐使用。
    - per_tensor量化粒度暂不支持。
    - 权重必须为2D张量。

## 功能介绍

### YAML配置示例

per_channel量化示例

```yaml
spec:
  process:
    - type: "linear_quant"
      qconfig:
        weight:
          scope: "per_channel"   # 量化范围
          dtype: "int8"          # 量化数据类型
          symmetric: true        # 是否对称量化
          method: "gptq"         # 量化算法-GPTQ
          ext: # 可选，扩展参数
            percdamp: 0.01       # 可选，阻尼系数，默认值0.01
            block_size: 128      # 可选，分块大小，默认值128
```

per_group量化示例

```yaml
spec:
  process:
    - type: "linear_quant"
      qconfig:
        weight:
          scope: "per_group"   # 量化范围
          dtype: "int8"        # 量化数据类型
          symmetric: true      # 是否对称量化
          method: "gptq"       # 量化算法-GPTQ
          ext: # 可选，扩展参数
            percdamp: 0.01     # 可选，阻尼系数，默认值0.01
            block_size: 128    # 可选，分块大小，默认值128
            group_size: 256    # 可选，分组大小，默认值256
```

### YAML配置字段详解

| 参数名       | 作用     | 可选值                            | 说明                                            | 默认值                         |
|-----------|--------|--------------------------------|-----------------------------------------------|-----------------------------|
| scope     | 量化范围   | `"per_channel"`, `"per_group"` | per_channel: 每个通道独立参数<br/>per_group: 每个分组独立参数 | `"per_channel"`             |
| dtype     | 量化数据类型 | `"int8"`                       | 8位整数量化                                        | `"int8"`                    |
| symmetric | 是否对称量化 | `true`, `false`                | true: 对称量化，零点为0<br/>false: 非对称量化，零点可调整        | `true`                      |
| method    | 量化方法   | `"gptq"`                       | gptq: gptq权重量化                                | `"gptq"`                    |
| ext       | 扩展配置   | `object`                       | 包含GPTQ特有的配置参数                                 | 见下方 **ext (GPTQ扩展配置)** 详细配置 |

**ext (GPTQ扩展配置)**

**作用**: 配置GPTQ算法特有的参数。

| 参数名        | 作用     | 类型      | 说明                                           | 示例值       |
|------------|--------|---------|----------------------------------------------|-----------|
| percdamp   | 阻尼系数   | `float` | percdamp 用于平滑梯度更新，减少量化引入的噪声对训练的影响            | 默认值`0.01` |
| block_size | 迭代分块大小 | `int`   | 分组量化的大小，必须能被待量化nn.Linear层的out_features维度整除   | 默认值`128`  |
| group_size | 量化分组大小 | `int`   | 分组量化的大小，必须能被待量化nn.Linear层的input_features维度整除 | 默认值`256`  |

## 模型适配

### 接口与数据结构

```python
# GPTQ量化器类
class WeightPerChannelGPTQ(AutoWeightQuantizer):
    def __init__(self, config: QConfig): ...

    def forward(self, x: Optional[torch.Tensor] = None) -> torch.Tensor: ...

    def init_weight(self, weight: QStorage, bias: Optional[torch.Tensor] = None) -> None: ...

    def get_q_storage(self) -> QStorage: ...

    def get_q_param(self) -> QParam: ...
```

### 适配步骤

- **前置要求**：
    - 权重必须为2D张量（如线性层的权重）。
    - 需要提供正确的量化配置（dtype、scope、method、symmetric）。
- **步骤**：
    1. 创建GPTQ量化配置：指定量化数据类型、范围、方法和对称性。
    2. 创建量化器实例：使用配置初始化WeightPerChannelGPTQ。
    3. 初始化权重：调用init_weight方法设置待量化的权重。
    4. 计算海森矩阵：调用forward方法收集激活值信息，计算hessian矩阵。
    5. 量化结果：通过get_q_storage和get_q_param触发权重量化，返回量化权重和量化参数。
