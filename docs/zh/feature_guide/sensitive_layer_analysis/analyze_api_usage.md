---
toc_depth: 3
---
# 量化敏感层分析工具使用指南

## 简介

`analyze` 是 msModelSlim 工具中的量化敏感层分析功能接口，用于分析模型中各层的量化敏感度，帮助用户识别量化敏感层，从而进行针对性的优化。

## 使用前准备

安装 msModelSlim 工具，详情请参见[《msModelSlim工具安装指南》](../../getting_started/install_guide.md)。

## 功能介绍

### 昇腾AI处理器支持情况

目前仅支持Atlas A3 训练系列产品/Atlas A3 推理系列产品和Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件，且内存需要大于1.5倍模型大小。

### 功能说明

- **多维度分析**: 支持 `std`、`quantile`、`kurtosis` 三种**线性层**衡量算法，以及 `attention_mse` 一种**attention结构**衡量算法，能够从数据分布、稳健性、峰态特征和量化前后注意力输出的差异等多个维度，精准评估层敏感度。
- **灵活配置**: 支持自定义校准数据集（JSON/JSONL格式）、层名匹配以及丰富的参数选项，满足不同场景的量化需求。
- **智能输出**: 支持打印Top K敏感层列表，实际打印数量可能会大于或等于目标数量，如QKV一起打印。

### 注意事项

- transformer版本依赖于模型，与量化功能无关。
- 实际回退的层数受推理引擎实现的限制，因此可能与topk参数设置存在一些差异。
- topk默认值为15，作为回退经验值仅供参考，如果打印层涉及qkv，会将qkv一起输出。
- 由于安全规范，trust_remote_code默认为False。
- 敏感层分析目前仅支持大语言模型。

### 命令格式

```bash
msmodelslim analyze [参数选项]
```

### 参数说明

#### 必需参数

| 参数 | 类型 | 默认值 | 描述 | 示例值 |
|------|------|--------|------|--------|
| `--model_type` | `str` | - | 模型类型，用于指定要分析的模型架构，见[参数详细说明](#参数详细说明) | `Qwen2.5-7B-Instruct` |
| `--model_path` | `str` | - | 原始模型的路径，建议使用绝对路径 | `/path/Qwen2.5-7B-Instruct` |

#### 可选参数

| 参数 | 类型 | 默认值 | 描述 | 示例值 |
|------|------|--------|------|--------|
| `--device` | `str` | `npu` | 指定运行分析的目标设备，可选值：`npu`, `cpu`。 | `npu` |
| `--pattern` | `List[str]` | `["*"]` | 待分析的层名称列表，支持通配符匹配。支持设置多个pattern，使用空格分隔。不传值会使用默认值。 | `"*linear*"` `"*attention.*"` `"*mlp.*"` |
| `--metrics` | `str` | `"kurtosis"` | 分析使用的度量算法，可选值：`"std"`, `"quantile"`, `"kurtosis"`, `"attention_mse"` | `"kurtosis"` |
| `--calib_dataset` | `str` | `"boolq.jsonl"` | 校准数据集文件路径，支持JSON/JSONL格式，以.json或.jsonl结尾。支持绝对路径和相对路径。 |`/path/data.jsonl`|
| `--topk` | `int` | `15` | 输出Top K敏感的层数量，为大于0的整数。推荐范围为10~20。 |  `15` |
| `--trust_remote_code` | `bool` | `False` | 是否信任远程代码，需要用户自行保障安全性。可选值：`True`, `False`。模型加载时如需依赖 transformers 库外的文件，则需指定 `--trust_remote_code` 为 `True`，如 DeepSeek-V3 系列模型。 | `False` |
| `-h, --help` | - | - | 命令行参数帮助信息 | - |

### 参数详细说明

#### model_type 支持列表

不同分析指标（metrics）所支持的 model_type 不同，请按所选指标查看对应列表。

**线性层衡量算法（std / quantile / kurtosis）支持的 model_type**

线性层衡量算法当前支持的 model_type 与 ModelslimV1 量化一致，具体可参阅 [config.ini](../../../../config/config.ini) 中的 `[ModelAdapter]` 。

**attention结构衡量算法（attention_mse）支持的 model_type**

| model_type       |
| ---------------- |
| DeepSeek-V3      |
| DeepSeek-V3-0324 |
| DeepSeek-R1      |
| DeepSeek-R1-0528 |
| DeepSeek-V3.1    |

- **不支持的 model_type**：若输入的 model_type 不在所选指标对应的列表中，系统会打印 warning 并可能回退到默认的模型适配器，或（如 attention_mse）因适配器未实现接口而报错。
- **建议**：使用上述列表中与 `--metrics` 匹配的 model_type，以获得正确的分析行为和兼容性。

#### 路径和文件要求

- **model_path**: 路径必须真实存在，建议使用绝对路径，必须包含有效的模型文件。
- **calib_dataset**:
  - 支持JSON格式，JSON文件内是一个字符串列表，可参考lab_calib目录下的anti_prompt.json。
  - 支持JSONL格式，每行一个JSON对象，可参考lab_calib目录下的jsonl格式，例如boolq.jsonl。
  - 相对路径在`lab_calib`目录中查找所配置的calib_dataset。

#### 算法选择说明

- **std**: 标准差算法，计算简单且性能好，适合常规场景。
- **quantile**: 分位数算法，对异常值不敏感，适合精度要求较高的场景。
- **kurtosis**: 峰度算法，能识别分布形态特征，适合需要精细控制的场景。
- **attention_mse**: 注意力 MSE 算法，基于量化前后两次前向的注意力输出差异，适合需要评估注意力模块量化的场景。

### 分析算法说明

#### std (Standard Deviation) - 标准差算法

##### 算法原理

- 计算激活值的最大值、最小值和标准差。
- 使用公式: `score = max(|max_value|, |min_value|) / std`。
- 反映数据的变异程度和范围。

##### 适用场景

- **推荐用于**: 常规量化场景。
- **优势**:
  - 计算简单且性能好。
  - 对数据分布变化敏感。
  - 直观反映数据波动性。
- **特点**:
  - 标准差越大，score越小（表示该层对量化更不敏感）。
  - 数据范围越大，score越大（表示该层对量化更敏感）。

#### quantile (Quantile-based) - 分位数算法

##### 算法原理

- 计算激活值的第1/4和第3/4分位数。
- 使用公式: `score = 2 * max_abs / 254 / (Q3 - Q1)`。
- 基于四分位距(IQR)评估分布特征。

##### 适用场景

- **推荐用于**: 需要考虑数据分布尾部的场景。
- **优势**:
  - 对异常值不敏感。
  - 反映数据分布的稳健性。
  - 适合量化精度要求较高的场景。
- **特点**:
  - IQR越大，score越小（表示分布越分散，对量化越不敏感）。
  - 数据绝对值越大，score越大（表示该层对量化越敏感）。

#### kurtosis (Kurtosis-based) - 峰度算法

##### 算法原理

- 计算激活值的峰度(Kurtosis)。
- 峰度公式: `kurtosis = E[(X-μ)**4] / σ**4 - 3`。
- 反映数据分布的峰态特征。

##### 适用场景

- **推荐用于**: 需要精准识别激活值分布中极端值的场景。
- **优势**:
  - 能识别分布的尖峰程度。
  - 对极端值敏感。
  - 适用于需要精细控制的量化场景。
- **特点**:
  - 峰度值越大，score越大，表示分布越集中，对量化越敏感。
  - 峰度值越接近0，score越小，表示分布接近正态，越不敏感。

#### attention_mse (Attention MSE) - 注意力 MSE 算法

##### 算法原理

- 分别采集模型量化前（浮点权重）和量化后（量化权重）注意力模块的激活输出
- 对同一层的浮点输出与量化输出计算MSE指标。
- 计算公式：$\text{MSE} = \frac{1}{n} \sum_{i=1}^{n} (y_{\text{float}}^{(i)} - y_{\text{quant}}^{(i)})^2$
- 反映注意力模块对量化的敏感性。
  
##### 适用场景

- **推荐用于**: 需要对注意力模块进行量化的场景。
- **前提条件**:
  - 所用 `model_type` 对应的模型适配器需继承并实现实现 **AttentionMSEAnalysisInterface**。

##### 适配流程

```python
class AttentionMSEAnalysisInterface(ABC):
    @abstractmethod
    def get_attention_module_cls(self) -> str:
        ...

    @abstractmethod
    def get_attention_output_extractor(self) -> Callable[[Union[tuple, torch.Tensor]], torch.Tensor]:
        ...
```

- **get_attention_module_cls()**：返回用于匹配注意力层的模块类名（字符串）。分析时会在该类型模块上挂 hook，仅对此类模块采集输出并计算 MSE。
- **get_attention_output_extractor()**：返回一个提取函数，入参为 attention 模块 `forward` 的完整输出（多为 `tuple` 或单个 `Tensor`），从该输出中取出用于敏感度分析的张量并返回。

##### 特点

- 数值越大表示该注意力层在量化后输出偏差越大、越敏感。

### 使用示例

#### 基本使用

```bash
# 指定分析算法和数据集
# model_path表示模型路径
# calib_dataset表示校准集路径
msmodelslim analyze \
    --model_type Qwen2.5-7B-Instruct \
    --model_path ${model_path} \
    --metrics quantile \
    --calib_dataset ${calib_dataset} \
    --topk 20 \
    --device cpu
```

#### 自定义层模式

```bash
# 只分析注意力层和MLP层
msmodelslim analyze \
    --model_type Qwen2.5-7B-Instruct \
    --model_path ${model_path} \
    --pattern "*attention*" "*mlp*" \
    --metrics std
```

#### 完整配置示例

```bash
msmodelslim analyze \
    --model_type Qwen3-32B \
    --model_path ${model_path} \
    --device npu \
    --pattern "*.down_proj*" "*.o_proj*"\
    --metrics kurtosis \
    --calib_dataset ${calib_dataset} \
    --topk 15 \
    --trust_remote_code False
```

### 输出说明

支持纯净的yaml格式打印，方便用户直接粘贴到yaml文件中。

#### 控制台输出

以上面的完整配置为例，控制台输出如下：

```text
...
msmodelslim.core.analysis_service.pipeline_analysis.service - INFO - ==========ANALYSIS: Starting Layer Analysis==========
msmodelslim.core.analysis_service.pipeline_analysis.service - INFO - Analysis metrics: kurtosis
msmodelslim.core.analysis_service.pipeline_analysis.service - INFO - Layer patterns: ['*.down_proj*', '*.o_proj*']
msmodelslim.core.runner.layer_wise_runner - INFO - Start to handle dataset
msmodelslim.core.runner.layer_wise_runner - INFO - Handle dataset success
msmodelslim.core.runner.layer_wise_runner - INFO - Start to init model
msmodelslim.core.runner.layer_wise_runner - INFO - Init model success
msmodelslim.core.runner.layer_wise_runner - INFO - KV cache requirement: False
msmodelslim.core.runner.layer_wise_runner - INFO - Scheduler 3 unit: [LoadProcessor(device=npu:0, non_blocking=False), UnaryAnalysisProcessor, LoadProcessor(device=meta, non_blocking=False)]
msmodelslim.core.runner.layer_wise_runner - INFO - Run processor LoadProcessor(device=npu:0, non_blocking=False) for "model.layers.0"
msmodelslim.core.runner.layer_wise_runner - INFO - Run processor UnaryAnalysisProcessor for "model.layers.0"
msmodelslim.core.runner.layer_wise_runner - INFO - Run processor LoadProcessor(device=meta, non_blocking=False) for "model.layers.0"
...
msmodelslim.app.analysis.application - INFO - === Layer Analysis Results (kurtosis method) ===
msmodelslim.app.analysis.application - INFO - Patterns analyzed: ['*.down_proj*', '*.o_proj*']
msmodelslim.app.analysis.application - INFO - Total layers analyzed: 128
msmodelslim.app.analysis.application - INFO - Layer Sensitivity Scores (higher score = more sensitive to quantization):
...
msmodelslim.app.analysis.application - INFO - === YAML Format for quantization ===
...
msmodelslim.app.analysis.application - INFO - === End of YAML Format ===
msmodelslim.app.analysis.application - INFO - ===========ANALYSIS COMPLETE===========
```

## FAQ

### 问题现象：校准数据集文件格式错误或无法读取

**解决方案**：

1. 确认文件格式为支持的JSON或JSONL格式。
2. 确保每条记录包含必要的字段。
3. 验证文件路径是否正确。
4. 确认校准集文件存在可读权限。

### 问题现象：输入不支持的model_type会发生什么？

**解决方案**：
当输入的model_type不在支持列表中时：

- 系统会打印warning日志，提示使用默认模型。
- 自动使用默认模型进行处理。
- 可能无法获得最佳的分析效果。
- **建议**：优先使用[model_type 支持列表](#model_type-支持列表)中的标准model_type，以获得最佳兼容性和分析精度。

### 问题现象：分析结果如何应用到量化配置中？

**解决方案**：

1. 将高敏感度层加入量化禁用列表，避免这些层被量化。
2. 对中等敏感度层使用较低的量化精度（如8bit替代4bit）。
3. 根据分析结果调整量化策略，实现精度与性能的最佳平衡。
