# PDMIX：激活值阶段间混合量化算法说明

## 简介

- 来源：华为自研。
- 问题：传统 W8A8 静态量化采用静态的激活量化参数，在长上下文或分布漂移场景中易产生较大量化误差，需要回退大量层才能控制精度损失，却因此损失性能收益。
- 目标：阶段间混合 W8A8 动态与 W8A8 静态两种策略，回退少量层即可控制精度损失，输出时获取与W8A8静态量化相近的性能收益：
    - prefilling 阶段：W8A8 动态量化（per-token），减少输入上下文的量化信息损失，控制量化精度损失；
    - decoding 阶段：W8A8 静态量化（per-tensor），获取输出时的量化性能收益，提高推理性能。

> 说明：阶段间权重量化方式必须保持一致，否则需要存储两份量化权重，因此 PDMIX 量化算法可视为一种`激活值量化`算法，与 W8
> per-channel 权重量化结合。

## 使用前准备

安装 msModelSlim 工具，详情请参见[安装指南](../install_guide.md)。

## 功能介绍

PDMIX 量化算法通过 ModelSlimV1 的 YAML 配置文件使用。

```yaml
spec:
  process:
    - type: "linear_quant"     # 线性层量化模式处理器
      qconfig:
        act: # 激活值量化配置
          scope: "pd_mix"      # prefilling: per_token；decoding: per_tensor
          dtype: "int8"        # 暂时仅支持 INT8
          symmetric: false     # PDMIX 量化总体为非对称
          method: "minmax"     # 暂时仅支持 MinMax 算法
        weight: # 权重量化配置   
          scope: "per_channel" # 暂时仅支持搭配权重 per-channel 量化
          dtype: "int8"        # 暂时仅支持搭配权重 INT8 量化
          symmetric: true      # 仅支持搭配权重对称量化
          method: "minmax"     # 权重量化算法
```

目前仅MindIE支持且只支持W8A8 PDMIX一种量化模式，因此量化配置除了`qconfig.weight.method`可调整外，其他配置组合均未有对应实现。

## 原理与实现

### 核心思想

- 保持权重量化不变，激活值量化采用“阶段自适应”的混合量化：
    - prefilling：per-token 动态量化，token 级颗粒度在线计算量化参数，减小量化误差；
    - decoding：per-tensor 静态量化，离线计算激活量化参数，减少量化参数计算操作，降低推理时延，提高吞吐量。

### 实现位置

- 量化校准：[`msmodelslim/quant/quantizer/impl/minmax.py`](../../../msmodelslim/quant/quantizer/impl/minmax.py) 中的
  `ActPDMixMinmax`
- 量化模式 IR：[`msmodelslim/quant/ir/w8a8_pdmix.py`](../../../msmodelslim/quant/ir/w8a8_pdmix.py) 中的
  `W8A8PDMixFakeQuantLinear`
- 相关常量：[`msmodelslim/quant/ir/const.py`](../../../msmodelslim/quant/ir/const.py) 定义 `int8_pd_mix_asym`

## 适用范围与局限性

- 当前仅支持MindIE推理部署
- 当前仅支持Atlas 800T A2、Atlas 800I A2、Atlas 800T A3、Atlas 800I A3系列产品推理部署
- 当静态量化精度损失大，需要回退的层较多时，可尝试替换为 PDMIX 量化
- 该算法为线性层量化算法中的激活值量化算法，线性层以 torch.nn.Linear 实现即满足算法需求


