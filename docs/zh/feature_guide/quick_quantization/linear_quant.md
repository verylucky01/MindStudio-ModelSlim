# LinearQuantProcess 线性层量化处理器

## 简介

LinearQuantProcess是modelslim_v1量化服务中的核心处理器，用于对模型的线性层进行量化处理。它支持灵活的量化配置，包括激活值量化和权重量化。

## 使用前准备

安装 msModelSlim 工具，详情请参见[安装指南](../../install_guide.md)。

## 功能介绍

### YAML配置示例

#### W8A8静态量化配置

```yaml
- type: "linear_quant"
  qconfig:
    act:
      scope: "per_tensor"
      dtype: "int8"
      symmetric: false
      method: "minmax"
    weight:
      scope: "per_channel"
      dtype: "int8"
      symmetric: true
      method: "minmax"
  include: [ "*" ]
  exclude: [ "*down_proj*" ]
```

#### W8A8动态量化配置

```yaml
- type: "linear_quant"
  qconfig:
    act:
      scope: "per_token"
      dtype: "int8"
      symmetric: false
      method: "minmax"
    weight:
      scope: "per_channel"
      dtype: "int8"
      symmetric: true
      method: "minmax"
  include: [ "*mlp*" ]
```

#### W4A8动态量化配置

```yaml
- type: "linear_quant"
  qconfig:
    act:
      scope: "per_token"
      dtype: "int8"
      symmetric: true
      method: "minmax"
    weight:
      scope: "per_channel"
      dtype: "int4"
      symmetric: true
      method: "minmax"
  include: [ "*" ]
```

### YAML配置字段详解

| 字段名 | 作用 | 类型 | 说明 | 示例值 |
|--------|------|------|------|--------|
| type | 处理器类型标识 | `string` | 固定值，用于标识这是一个线性层量化处理器 | `"linear_quant"` |
| qconfig | 量化配置参数 | `object` | 包含激活值量化和权重量化的详细配置 | 见下方详细配置 |
| include | 包含的层模式 | `array[string]` | 支持通配符匹配，指定要量化的层 | `["*"]`, `["*self_attn*"]` |
| exclude | 排除的层模式 | `array[string]` | 支持通配符匹配，优先级高于include | `["*down_proj*"]` |

### qconfig.act (激活值量化配置)

**作用**: 配置激活值的量化参数。

| 参数名 | 作用 | 可选值                                       | 说明                                                                                                               | 默认值 |
|--------|------|-------------------------------------------|------------------------------------------------------------------------------------------------------------------|--------|
| scope | 量化范围 | `"per_tensor"`, `"per_token"`, `"pd_mix"` | per_tensor: 整个张量使用相同参数<br/>per_token: 每个token独立参数（动态量化）<br/>pd_mix: prefilling时per_token，decoding时per_tensor | `"per_tensor"` |
| dtype | 量化数据类型 | `"int8"`, `"int4"`                        | 8位/4位整数量化                                                                                                        | `"int8"` |
| symmetric | 是否对称量化 | `true`, `false`                           | true: 对称量化，零点为0<br/>false: 非对称量化，零点可调整                                                                           | `false` |
| method | 量化方法 | `"minmax"`, `"histogram"`                 | minmax: 最小最大值量化<br/>histogram: 直方图量化                                                                             | `"minmax"` |

### qconfig.weight (权重量化配置)

**作用**: 配置权重的量化参数。

| 参数名 | 作用 | 可选值 | 说明 | 默认值 |
|--------|------|--------|------|--------|
| scope | 量化范围 | `"per_tensor"`, `"per_channel"` | per_tensor: 整个张量使用相同参数<br/>per_channel: 每个通道独立参数 | `"per_channel"` |
| dtype | 量化数据类型 | `"int8"`, `"int4"` | 8位/4位整数量化 | `"int8"` |
| symmetric | 是否对称量化 | `true`, `false` | true: 对称量化，零点为0<br/>false: 非对称量化，零点可调整 | `true` |
| method | 量化方法 | `"minmax"`, `"ssz"` | minmax: 最小最大值量化<br/>ssz: ssz权重量化 | `"minmax"` |

## 功能介绍

### 层过滤机制详解

#### 过滤规则

1. **include规则**: 定义要包含的层，只有匹配include模式的层才会被处理。
2. **exclude规则**: 定义要排除的层，匹配exclude模式的层会被跳过。
3. **优先级**: exclude规则的优先级高于include规则。

#### 匹配模式（Unix通配符的匹配模式）

| 通配符 | 作用 | 示例 |
|--------|------|------|
| `*` | 匹配任意字符序列 | `*self_attn*` 匹配包含 "self_attn" 的任意字符串 |
| `?` | 匹配单个字符 | `layer?` 匹配 "layer1", "layerA" 等 |
| `[abc]` | 匹配字符集中的任意一个字符 | `layer[123]` 匹配 "layer1", "layer2", "layer3" |
| `[!abc]` | 匹配不在字符集中的任意字符 | `layer[!123]` 匹配除 "layer1", "layer2", "layer3" 外的字符串 |

#### 过滤顺序

1. **第一步**: 检查层名是否匹配include模式。
    - 如果include为空或未设置，默认包含所有层。
    - 如果层名不匹配任何include模式，该层被排除。
2. **第二步**: 检查层名是否匹配exclude模式。
    - 如果层名匹配任何exclude模式，该层被排除。
    - 即使该层在第一步中被include包含，也会被exclude排除。

#### 示例说明

##### 示例1: 基础过滤

```yaml
include: [ "*" ]
exclude: [ "*down_proj*" ]
```

- **结果**: 包含所有层，但排除包含"down_proj"的层。

##### 示例2: 选择性包含

```yaml
include: [ "*self_attn*", "*mlp*" ]
exclude: [ ]
```

- **结果**: 只包含包含"self_attn"或"mlp"的层。

##### 示例3: 复杂过滤

```yaml
include: [ "*attention*", "*mlp*" ]
exclude: [ "*down_proj*", "*gate*" ]
```

- **结果**: 包含包含"attention"或"mlp"的层，但排除包含"down_proj"或"gate"的层。

##### 示例4: 精确匹配

```yaml
include: [ "model.layers.*.self_attn.*" ]
exclude: [ "model.layers.*.self_attn.down_proj" ]
```

- **结果**: 只包含self_attn层，但排除其中的down_proj子层。

#### 常见层名模式

#### Transformer架构常见层名

| 模式 | 描述 | 典型用途 |
|------|------|----------|
| `*self_attn*` | 自注意力层 | 注意力机制相关的权重和偏置 |
| `*mlp*` | 多层感知机层 | 前馈网络层 |
| `*attention*` | 注意力相关层 | 广义的注意力相关组件 |
| `*ffn*` | 前馈网络层 | Feed Forward Network |
| `*gate*` | 门控层 | 门控机制相关层 |
| `*down_proj*` | 下投影层 | 降维投影层 |
| `*up_proj*` | 上投影层 | 升维投影层 |

#### 量化策略建议

| 策略 | 配置 | 说明 |
|------|------|------|
| 全量量化 | `include: ["*"]` | 量化所有线性层 |
| 注意力层量化 | `include: ["*self_attn*"]` | 只量化自注意力相关层 |
| MLP层量化 | `include: ["*mlp*"]` | 只量化多层感知机层 |
| 排除敏感层 | `exclude: ["*down_proj*", "*gate*"]` | 排除对精度敏感的层 |

## FAQ

### 量化组合有效性

**重要提醒**: 并非所有配置组合都是有效的量化组合，无效组合会抛出`UnsupportedError`异常。

#### 错误处理

- 当检测到无效的量化配置组合时，工具会抛出`UnsupportedError`异常。
- 异常信息会详细说明具体的配置冲突原因。
- 请根据异常信息调整配置参数。

### 层匹配告警

**重要提醒**: 当include/exclude模式未匹配到任何层时，工具会进行告警，请务必关注这些告警信息。

#### 常见匹配失败原因

1. **层名不匹配**
   ```yaml
   # ❌ 可能匹配失败：层名中不包含"self_attn"
   include: ["*self_attn*"]
   # 实际层名可能是: "attention", "attn", "self_attention"
   ```

2. **路径层级错误**
   ```yaml
   # ❌ 可能匹配失败：路径层级不匹配
   include: ["model.layers.*.attention"]
   # 实际路径可能是: "layers.*.attention" 或 "transformer.layers.*.attention"
   ```

3. **大小写敏感**
   ```yaml
   # ❌ 可能匹配失败：大小写不匹配
   include: ["*SelfAttn*"]
   # 实际层名可能是: "*self_attn*"
   ```

4. **拼写错误**
   ```yaml
   # ❌ 可能匹配失败：拼写错误
   include: ["*self_atttn*"]  # 拼写错误：多了一个t
   # 实际层名可能是: "*self_attention*"
   
   include: ["*mlp*"]  # 可能匹配失败：不同模型使用不同命名
   # 实际层名可能是: "*ffn*" 或 "*feed_forward*"
   ```

5. **不是nn.Linear的路径**
   ```yaml
   # ❌ 可能匹配失败：匹配到非Linear层
   include: ["*gate.weight"]
   # 实际上是nn.Parameters的路径
   # LinearQuantProcess只处理nn.Linear层，其他层会被忽略
   ```

#### 排查步骤

1. **检查层名**: 使用模型分析工具查看实际的层名结构。
2. **验证模式**: 使用简单的通配符模式进行测试。
3. **逐步调试**: 从`include: ["*"]`开始，逐步缩小范围。
4. **查看日志**: 关注工具输出的匹配结果和告警信息。
    - 匹配失败告警: `patterns are not matched any module, please ensure this is as expected`。
