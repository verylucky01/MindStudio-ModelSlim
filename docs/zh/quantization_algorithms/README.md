---
toc_depth: 3
---
# 量化算法总览

msModelSlim 支持多种先进的量化算法，涵盖了从离群值抑制到低比特优化的各个环节。下表按类别总结了目前支持的核心算法及其主要特性。

## 离群值抑制算法
离群值抑制算法旨在平滑激活值的分布，减少量化带来的精度损失。

| 算法名称 | 核心思想 | 适用场景 | 详细说明 |
| :--- | :--- | :--- | :--- |
| **QuaRot** | 应用正交旋转矩阵平滑激活值分布 | 抑制激活离群值，提升精度 | [查看详情](outlier_suppression_algorithms/quarot.md) |
| **SmoothQuant** | 协同缩放激活与权重，平滑离群值 | 抑制激活离群值 | [查看详情](outlier_suppression_algorithms/smooth_quant.md) |
| **Iterative Smooth** | 迭代式平滑缩放，更精细的分布调整 | 复杂分布下的精度优化 | [查看详情](outlier_suppression_algorithms/iterative_smooth.md) |
| **Flex Smooth Quant** | 二阶段网格搜索自动寻找最优 alpha/beta | 灵活适配不同架构 | [查看详情](outlier_suppression_algorithms/flex_smooth_quant.md) |
| **Flex AWQ SSZ** | 结合 AWQ 与 SSZ，使用真实量化器评估误差 | 自动搜索最优平滑参数 | [查看详情](outlier_suppression_algorithms/flex_awq_ssz.md) |
| **KV Smooth** | 针对 KV Cache 的平滑抑制算法 | 降低 KV Cache 显存占用 | [查看详情](outlier_suppression_algorithms/kv_smooth.md) |

## 量化算法
包含权重量化、激活量化以及针对特定结构的量化方案。

| 算法名称 | 类型 | 核心思想 | 适用场景 | 详细说明 |
| :--- | :--- | :--- | :--- | :--- |
| **AutoRound** | 权重量化优化 | 基于 SignSGD 优化舍入偏移，降低重构误差 | 4bit 等超低比特量化 | [查看详情](quantization_algorithms/autoround.md) |
| **FA3 Quant** | 激活量化 | 针对 Attention 激活的 per-head INT8 量化 | 长序列、MLA 架构模型 | [查看详情](quantization_algorithms/fa3_quant.md) |
| **GPTQ** | 权重量化优化 | 通过逐列优化和误差补偿最小化量化误差 | 高精度权重量化需求 | [查看详情](quantization_algorithms/gptq.md) |
| **KVCache Quant** | KV Cache 量化 | 针对 KV Cache 的量化方案 | 提升长序列推理效率 | [查看详情](quantization_algorithms/kvcache_quant.md) |
| **Linear Quant** | 基础量化 | 对线性层进行权重量化和激活量化 | 基础量化场景 | [查看详情](quantization_algorithms/linear_quant.md) |
| **PDMIX** | 混合阶段量化 | Prefilling 使用动态量化，Decoding 使用静态量化 | 大模型推理加速，平衡精度与性能 | [查看详情](quantization_algorithms/pdmix.md) |
| **Histogram** | 激活量化 | 分析直方图分布，搜索最优截断区间 | 过滤离群值，提高精度 | [查看详情](quantization_algorithms/histogram_activation_quantization.md) |
| **MinMax** | 基础量化 | 统计最大最小值确定量化范围 | 基础量化场景，计算开销低 | [查看详情](quantization_algorithms/minmax.md) |
| **SSZ** | 权重量化 | 迭代搜索最优缩放因子和偏移量 | 权重分布不均的精度优化 | [查看详情](quantization_algorithms/ssz.md) |
| **LAOS** | 低比特量化 | 针对 W4A4 等极低比特场景的优化 | 极致压缩需求 | [查看详情](quantization_algorithms/laos.md) |
| **Float Sparse** | 稀疏化 | 基于 ADMM 算法实现模型浮点 sparse | 高压缩率需求 | [查看详情](quantization_algorithms/float_sparse.md) |

## 自动调优策略
通过自动化策略寻找最优的量化配置。

| 算法名称 | 核心思想 | 适用场景 | 详细说明 |
| :--- | :--- | :--- | :--- |
| **Standing High** | 针对特定层的精度保护策略 | 解决关键层量化损失 | [查看详情](auto_tuning_strategies/standing_high.md) |

## 敏感层分析
评估模型各层对量化的敏感程度，辅助决定精度保护策略。

| 算法名称 | 核心思想 | 适用场景 | 详细说明 |
| :--- | :--- | :--- | :--- |
| **std** | 基于标准差评估数据的变异程度与量程关系 | 常规量化场景，首选基准评估方法 | [查看详情](sensitive_layer_analysis/algorithms.md#1-std-standard-deviation---标准差算法) |
| **quantile** | 基于分位数和四分位距（IQR）评估分布稳健性 | 数据分布存在长尾效应或极端离群值 | [查看详情](sensitive_layer_analysis/algorithms.md#2-quantile-quantile-based---分位数算法) |
| **kurtosis** | 基于峰度衡量分布的尖锐程度和尾部厚度 | 识别极端值影响，追求极致精度的场景 | [查看详情](sensitive_layer_analysis/algorithms.md#3-kurtosis-kurtosis-based---峰度算法) |


## 算法选择建议

- **初学者**：建议优先使用 [一键量化 (V1)](../feature_guide/quick_quantization_v1/usage.md)，它会自动集成合适的算法组合。
- **追求极致精度**：可以尝试组合使用 **QuaRot** + **AutoRound**。
- **长序列推理**：推荐开启 **FA3 Quant** 和 **KVCache Quant**。
