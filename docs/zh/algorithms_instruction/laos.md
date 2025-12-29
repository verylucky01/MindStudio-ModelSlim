# LAOS：w4a4量化方案说明

## 简介

- **来源**：华为自研。
- **背景**：在低比特量化（如W4A4）场景下，模型精度损失尤为显著，其核心难点在于权重和激活值中的极端离群值会显著扭曲量化区间，导致数值表示精度急剧下降，传统方法难以解决。
- **核心思想**：核心思想是“协同优化”。通过 [QuaRot](quarot.md) 和 [Iterative Smooth](iterative_smooth.md) 技术对激活分布进行平滑处理，有效抑制离群值，为后续量化创造良好条件；再利用 [AutoRound](autoround.md) 自适应功能为不同权重确定最优舍入策略，从而提高大模型在低比特量化场景的精度。

## 使用前准备

安装 msModelSlim 工具，详情请参见[安装指南](../install_guide.md)。

## 功能介绍

## 昇腾AI处理器支持情况

| 产品系列 | 支持 |
|---------|------|
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 | ✓ |
| Atlas A2 训练系列产品/Atlas 800I A2 推理产品 | ✓ |
| Atlas 推理系列产品 | ✗ |

**注：算法实现包含训练过程，对NPU显存有一定的要求，仅支持NPU显存>=64G的设备。**

### 修改配置文件使用

```yaml
apiversion: modelslim_v1

default_w8a8_dynamic: &default_w8a8_dynamic
  weight:
    scope: "per_group"
    dtype: "int8"
    symmetric: True
    method: "autoround"
    ext:
      group_size: 256
      scale_dtype: "bfloat16"
  act:
    scope: "per_token"
    dtype: "int8"
    symmetric: True
    method: "minmax"


default_w4a4_dynamic: &default_w4a4_dynamic
  weight:
    scope: "per_group"
    dtype: "int4"
    symmetric: True
    method: "autoround"
    ext:
      group_size: 256
      scale_dtype: "bfloat16"
  act:
    scope: "per_token"
    dtype: "int4"
    symmetric: True
    method: "minmax"


spec:
  process:
    - type: "iter_smooth"
      alpha: 0.9
      scale_min: 1e-5
      symmetric: False
      enable_subgraph_type: [ "ov", "up-down" ]

    - type: "quarot"
      online: True
      block_size: 32
      max_tp_size: 4
      down_proj_online_layers: [ 1,3,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26 ]

    - type: "iter_smooth"
      alpha: 0.9
      scale_min: 1e-5
      symmetric: False
      enable_subgraph_type: [ "norm-linear" ]

    - type: "autoround_quant"
      iters: 400
      enable_minmax_tuning: True
      enable_round_tuning: True
      strategies:
        - qconfig: *default_w8a8_dynamic
          exclude:
            - "*.up_proj"
            - "*.gate_proj"
            - "*.o_proj"
            - "model.layers.{1,3,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26}.mlp.down_proj"

        - qconfig: *default_w4a4_dynamic
          include:
            - "*.up_proj"
            - "*.gate_proj"
            - "*.o_proj"
            - "model.layers.{1,3,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26}.mlp.down_proj"

  save:
    - type: "ascendv1_saver"
      part_file_size: 4

  dataset: boolq.jsonl

```

### YAML配置字段详解

配置字段来自Iterative Smooth、QuaRot、AutoRound三者结合，详见[Iterative Smooth YAML配置字段详解](iterative_smooth.md#yaml配置字段详解)、[QuaRot YAML配置字段详解](quarot.md#yaml配置字段详解)、[AutoRound YAML配置字段详解](autoround.md#yaml配置字段详解)。

## 模型适配

### 适配步骤

- **前置要求**：

  - 确保所有返回的模块引用都是实际模型中的模块对象。
  - 模块路径必须与model.named_modules()返回的路径完全一致。

- **步骤**：

  1. 在配置文件中定义量化策略，支持针对不同的层使用不同的量化策略。
  2. 在配置文件中使用"iter_smooth"和"quarot"进行离群值抑制，需要注意二者使用的顺序，具体请参见配置文件。
  3. 在配置文件中使用"autoround_quant"指定autoround处理器，并且配置相关参数。
  4. 如需使用自定义校准集，可参考 `msmodelslim/lab_calib`添加数据集，并在配置文件中指定数据集名称。

### 适用范围和局限性

- **低比特量化**：适合极低比特量化场景中的4比特量化。
- **高精度需求**：在低比特条件下仍能保持较高的模型精度。
- **计算资源**：需要额外的优化过程，计算成本高于简单量化方法。
- **使用限制**：
  - 需要足够的校准数据或训练迭代次数来优化参数。
  - 受限于QuaRot算法的接口实现，当前该方案仅支持Qwen3稠密系列模型（如Qwen3-8B/14B/32B），未具备泛化至其他系列模型的能力，具体限制请参考 [QuaRot适用范围与局限性](quarot.md#适用范围与局限性)。
  - 若在QuaRot配置中启用了在线旋转，在使用推理引擎以TP并行的方式进行部署时，需要保证`tp_size`需要小于等于QuaRot的配置参数`max_tp_size`，否则必然导致精度异常，具体限制请参考 [QuaRot适用范围与局限性](quarot.md#适用范围与局限性)。

## 常见问题排查

### 1. 多种离群值抑制方法配合问题

- **现象**：量化配置的`process`配置中，`quarot`和`iter_smooth`的顺序设置不当，会产生互斥效应，影响方案执行的正确性。
- **解决方案**：严格参考示例中的离群值配置顺序进行配置。
