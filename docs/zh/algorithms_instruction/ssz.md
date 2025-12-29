# SSZ：权重量化算法说明

## 简介

- **来源**：华为自研。
- **问题**：传统量化方法（如MinMax）在权重分布不均匀时，量化误差较大，影响模型精度。
- **目标**：通过迭代搜索最优的缩放因子（scale）和偏移量（offset）来最小化量化误差，提高量化模型的精度。

## 使用前准备

安装 msModelSlim 工具，详情请参见[安装指南](../install_guide.md)。

## 功能介绍

作为量化器使用：

```python
from msmodelslim.quant.quantizer.base import QConfig
from msmodelslim.quant.quantizer.impl.ssz import WeightPerChannelSsz

# 创建SSZ量化配置
config = QConfig(
    dtype="int8",           # 量化数据类型
    scope="per_channel",    # 量化范围：per_channel
    method="ssz",          # 量化方法：ssz
    symmetric=true         # 对称量化
)

# 创建量化器
quantizer = WeightPerChannelSsz(config)
```

### YAML配置示例

```yaml
spec:
  process:
    - type: "linear_quant" 
      qconfig:
        weight:
          scope: "per_channel"
          dtype: "int8" 
          symmetric: true
          method: "ssz"
```

### YAML配置字段详解

| 参数名 | 作用 | 可选值 | 说明 | 默认值 |
|--------|------|--------|------|--------|
| scope | 量化范围 | `"per_tensor"`, `"per_channel"` | per_tensor: 整个张量使用相同参数<br/>per_channel: 每个通道独立参数 | `"per_channel"` |
| dtype | 量化数据类型 | `"int8"`, `"int4"` | 8位/4位整数量化 | `"int8"` |
| symmetric | 是否对称量化 | `true`, `false` | true: 对称量化，零点为0<br/>false: 非对称量化，零点可调整 | `true` |
| method | 量化方法 | `"ssz"` | ssz: ssz权重量化 | `"ssz"` |

## 原理和实现

### 原理

SSZ算法基于以下核心思想：

1. **迭代优化**：通过多次迭代来逐步优化量化参数。
2. **最小二乘法**：使用最小二乘法计算当前最优的 scale 和 offset。
3. **贪心更新**：只保留能改善量化误差的参数。
4. **收敛判断**：通过相对和绝对误差变化来判断收敛。

算法流程：
```
1. 使用 MinMax 观察器初始量化参数 scale 和 offset。
2. 通过最小二乘法计算当前最优的 scale 和 offset。
3. 比较新旧参数的量化误差，保留更好的参数。
4. 重复步骤 2-3 直到收敛或达到最大迭代次数，得到最终的量化参数。
```

### 实现

- 算法在 `msmodelslim/quant/quantizer/impl/ssz.py` 中实现，核心函数为 `ssz_calculate_qparam`：
    1. **初始化阶段**：
        - 使用MinMax观察器计算权重的统计信息（min/max值）。
        - 基于统计信息计算初始的量化参数（scale和offset）。
    2. **迭代优化阶段**：
        - 对称量化：offset固定为0，只优化scale。
        - 非对称量化：同时优化scale和offset。
        - 使用最小二乘法计算最优参数。
        - 贪心更新策略：只保留能改善量化误差的参数。
    3. **收敛判断**：
        - 相对误差变化：`(best_mse - current_mse) / best_mse < threshold`。
        - 绝对误差变化：`|best_mse - current_mse| < threshold`。
        - 所有通道都满足收敛条件时提前退出。

## 模型适配

### 接口与数据结构

```python
# SSZ量化器类
class WeightPerChannelSsz(AutoWeightQuantizer):
    def __init__(self, config: QConfig): ...
    
    def forward(self, x: Optional[torch.Tensor] = None) -> torch.Tensor: ...
    
    def init_weight(self, weight: QStorage, bias: Optional[torch.Tensor] = None) -> None: ...
    
    def get_q_storage(self) -> QStorage: ...
    
    def get_q_param(self) -> QParam: ...

# 核心算法函数
def ssz_calculate_qparam(weight: QStorage, q_param: QParam) -> QParam: ...
```

### 适配步骤

- **前置要求**：
    - 权重必须是2D张量（如线性层的权重）。
    - 需要提供正确的量化配置（dtype、scope、method、symmetric）。
- **步骤**：
    1. 创建SSZ量化配置：指定量化数据类型、范围、方法和对称性。
    2. 创建量化器实例：使用配置初始化WeightPerChannelSsz。
    3. 初始化权重：调用init_weight方法设置待量化的权重。
    4. 执行量化：调用forward方法进行量化计算。
    5. 获取结果：通过get_q_storage和get_q_param获取量化结果。

### 完整示例

```python
import torch
from msmodelslim.quant.quantizer.base import QConfig, AutoWeightQuantizer
from msmodelslim.core.QAL.qbase import QStorage, QDType

# 1. 创建配置
config = QConfig(
    dtype="int8",
    scope="per_channel", 
    method="ssz",
    symmetric=True
)

# 2. 创建量化器
quantizer = AutoWeightQuantizer.from_config(config)

# 3. 准备权重数据
weight_tensor = torch.randn(256, 512)
weight_storage = QStorage(QDType.FLOAT, weight_tensor)

# 4. 初始化权重
quantizer.init_weight(weight_storage)

# 5. 执行量化
dequantized_weight = quantizer.forward()

# 6. 获取量化结果
q_storage = quantizer.get_q_storage()
q_param = quantizer.get_q_param()

print(f"原始权重形状: {weight_tensor.shape}")
print(f"量化后权重形状: {q_storage.value.shape}")
print(f"量化参数: {q_param}")
```

## 算法参数

SSZ算法内部使用以下参数（可通过修改源码调整）：

```python
SCALE_SEARCH_ITER_NUM = 20                    # 最大迭代次数
SCALE_SEARCH_CONVERGE_THRESHOLD = 1e-10       # 收敛阈值
SCALE_SEARCH_MIN_SCALE = 1e-5                 # 最小缩放因子
```

## 量化配置参数

```python
QConfig(
    dtype="int8",           # 量化数据类型：int8
    scope="per_channel",    # 量化范围：per_channel（每个通道独立量化）
    method="ssz",          # 量化方法：ssz
    symmetric=True         # 对称量化：True为对称，False为非对称
)
```

## 适用要求

- **高精度需求**：适用于对精度要求较高的模型量化场景。
- **权重分布不均匀**：特别适合权重分布不均匀的线性层。
- **计算成本**：SSZ算法需要多次迭代，某些场景下计算成本较大。
- **初始化依赖**：需要先使用MinMax观察器计算初始量化参数。
- **使用限制**：
    - 目前支持int8和int4场景的per_channel对称量化。
    - int4场景的per_channel非对称量化暂不支持（后续支持）。
    - per_tensor和per_group量化粒度暂不支持（后续支持）。
    - 权重必须是2D张量。

## 常见问题排查
### 1. 权重维度错误
**现象**：输入的权重维度错误，导致量化失败。
**解决方案**：检查权重维度是否正确，确保权重是2D张量。

### 2. 量化配置错误
**现象**：量化配置错误，导致量化失败。
**解决方案**：检查dtype、scope、method、symmetric参数设置是否正确。

### 3. 初始化顺序错误
**现象**：初始化顺序错误，导致量化失败。
**解决方案**：必须先调用init_weight，再调用forward。

### 4. 收敛问题
**现象**：如果算法不收敛，可以调整SCALE_SEARCH_CONVERGE_THRESHOLD参数。
**解决方案**：调整SCALE_SEARCH_CONVERGE_THRESHOLD参数。

