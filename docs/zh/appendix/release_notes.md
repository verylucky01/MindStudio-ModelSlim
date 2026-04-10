# 版本说明

## 版本配套说明

### 产品版本信息

| 产品名称        | 产品版本           | 版本类型 |
|-------------|----------------|------|
| msModelSlim | 26.0.0.alpha02 | 内测版本 |
| msModelSlim | 26.0.0.alpha01 | 内测版本 |
| msModelSlim | 8.3.0          | 正式版本 |

### 相关产品版本配套说明

| msModelSlim版本 | CANN版本 | PyTorch版本         | torch_npu版本    | Python版本         | Transformers版本 |
|---------------|--------|-------------------|----------------|------------------|----------------|
| 26.0.0.alpha02         | 不依赖特定版本 | 与具体模型有关，请参考相关模型资料 | 与具体模型有关，请参考相关模型资料 | Python 3.10、3.11 | 与具体模型有关，请参考[example](https://gitcode.com/Ascend/msmodelslim/tree/master/example)目录下对应模型的案例说明 |
| 26.0.0.alpha01         | 不依赖特定版本 | 与具体模型有关，请参考相关模型资料 | 与具体模型有关，请参考相关模型资料 | Python 3.10、3.11 | 与具体模型有关，请参考[example](https://gitcode.com/Ascend/msmodelslim/tree/master/example)目录下对应模型的案例说明 |
| 8.3.0         | 8.2.RC1及以上版本 | 与具体模型有关，请参考相关模型资料 | 与具体模型有关，请参考相关模型资料 | Python 3.10、3.11 | 与具体模型有关，请参考[example](https://gitcode.com/Ascend/msmodelslim/tree/master/example)目录下对应模型的案例说明 |

### whl包获取

|       版本        |     下载链接           |        校验码                 |
|:--------------:|:------------------------------------------------------------------------------------------------------------------------------------------------------:|:----------------------------------------------------------------:|
| 26.0.0-alpha.2 | [msmodelslim-26.0.0a2-py3-none-any.whl](https://gitcode.com/Ascend/msmodelslim/releases/download/tag_mindstudio_26.0.0.alpha02/msmodelslim-26.0.0a2-py3-none-any.whl) | 4711edb30c4354fcb99fb69a2e0351561b013bb1298d6f54a0ee409bf979a264 |
| 26.0.0-alpha.1 | [msmodelslim-26.0.0a1-py3-none-any.whl](https://gitcode.com/Ascend/msmodelslim/releases/download/tag_MindStudio_26.0.0-alpha.1/msmodelslim-26.0.0a1-py3-none-any.whl) | 60383c42bf103cf2f78304b3b974e2dac0190f0f20706a5ef347e55855048f42 |

更多详情可见 [release](https://gitcode.com/Ascend/msmodelslim/releases?presetConfig={%22tags%22:32,%22release%22:2)

## 版本兼容性说明

请参照上表了解各版本的兼容信息。

## 特性变更说明

### 26.0.0.alpha02

1. 支持通过 entry point 引入自定义 practice 目录，搭建model_adapter插件化能力基础
2. 优化自动调优功能
3. 支持Qwen3-Coder-480B模型W4A8量化、Qwen3.5 MOE模型W8A8量化
4. 支持GLM-4.7模型W8A8量化、GLM-5模型W4A8模型量化
5. 支持Qwen2.5-Omni-7B模型W8A8量化、Qwen3-Omni-30B-A3B模型W8A8量化

### 26.0.0.alpha01

1. 支持 Qwen3-VL-32B-Instruct W8A8 量化
2. 支持量化精度反馈自动调优，可根据精度需求自动搜索最优量化配置
3. 支持自主量化多模态理解模型，支持多模态理解模型的量化接入
4. 一键量化支持多卡量化，支持分布式逐层量化，提升大模型量化效率
5. 支持 DeepSeek-V3.2 W8A8 量化，单卡64G显存、100G内存即可执行
6. 支持 DeepSeek-V3.2-Exp W4A8 量化，单卡64G显存、100G内存即可执行
7. 支持 Qwen3-VL-235B-A22B W8A8 量化

### 8.3.0

1. DeepSeek-V3.2-Exp模型W8A8量化
2. DeepSeek-V3.1模型W8A8C8量化
3. Qwen3-32B模型W8A8C8量化
4. Qwen3-Next-80B模型W8A8量化
5. DeepSeek-R1-0528模型W4A8C8量化
