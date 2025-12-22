# 一键量化快速入门

## 概述

一键量化功能面向零基础用户，集成热门开源模型量化功能，具备“开箱即用”的特性。本功能支持全局调用量化命令，用户指定必要参数后，即可对目标原始权重执行指定的量化操作。
下面将以 Qwen2.5-7B-Instruct 为例进行介绍。

## 环境准备

### 1.安装msModelSlim

根据[安装指南](install_guide.md)完成开发环境配置。

### 2.下载大模型原始浮点权重

以 Qwen2.5-7B-Instruct 为例，可前往[Qwen2.5-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct)获取原始模型权重。

### 3.安装其他依赖（与模型相关）

```shell
pip install transformers==4.43.1
```

## 使用说明

一键量化功能通过命令行方式启动，正确安装 msModelSlim 工具后，可以通过如下命令运行：

```bash
msmodelslim quant [ARGS]
```

例如，使用一键量化功能量化 Qwen2.5-7B-Instruct 模型，量化方式采用 w8a8 ，则量化命令如下，其中\${MODEL_PATH}为Qwen2.5-7B-Instruct原始浮点权重路径，\${SAVE_PATH}为用户自定义的量化权重保存路径。详细接口说明请参考[一键量化接口说明](feature_guide/quick_quantization/usage.md#参数说明)。请注意`trust_remote_code`为`True`时可能执行浮点模型权重中代码文件，请确保浮点模型来源安全可靠。

```bash
msmodelslim quant --model_path ${MODEL_PATH} --save_path ${SAVE_PATH} --device npu --model_type Qwen2.5-7B-Instruct --quant_type w8a8 --trust_remote_code True
```

用户输入命令后，系统将根据指定需求，在最佳实践库中匹配到最佳配置从而实施量化。

**量化结果输出展示：**

```yaml
├── config.json                          # 原始模型配置文件
├── generation_config.json               # 原始生成配置文件  
├── quant_model_description.json         # 量化权重描述文件
├── quant_model_weight_w8a8.safetensors  # 量化权重文件
├── tokenizer_config.json                # 原始分词器配置文件
├── tokenizer.json                        # 原始分词器词汇表
└── vocab.json                            # 原始词汇映射文件
```

> **说明：** 
> - `quant_model_description.json` - 包含量化参数和配置信息。
> - `quant_model_weight_w8a8.safetensors` - 实际的量化模型权重（W8A8表示权重8位量化、激活8位量化）。
> - 其他文件为模型推理所需的配置和词汇表文件，来自原始浮点目录。

## 其他说明

### 一键量化支持矩阵

可通过[大模型支持矩阵](foundation_model_support_matrix.md)查看不同模型的一键量化支持情况，其中标记了`一键量化`的模型则已支持一键量化。

### 相关资料
- 对于过大的模型，可以参考[一键量化的逐层量化特性说明](./feature_guide/quick_quantization/layer_wise_quantization.md)使用逐层量化，能够明显降低显存使用。
- 对于一键量化支持的多种算法，可以参考[一键量化V1架构支持的算法](./algorithms_instruction)。