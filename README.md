<div align="center">

# msModelSlim

[![说明文档](https://img.shields.io/badge/Documentation-latest-brightgreen.svg?style=flat)](./docs/README.md)
[![license](https://img.shields.io/badge/License-Apache%202.0-blue)](../LICENSE)

[安装指南](./docs/安装指南.md) |
[快速入门](./docs/快速入门/一键量化快速入门.md) |
[支持矩阵](./docs/支持矩阵/大模型支持矩阵.md) |
[功能指南](./docs/README.md#功能指南) |
[自主量化](./docs/自主量化/模型接入.md) |
[案例集](./docs/README.md#案例集) |
[FAQ](./docs/FAQ.md)

</div>

## 🔥🔥🔥Latest News
- [2025/12/10] 🚀 msModelSlim 支持 [DeepSeek-V3.2-Exp W4A8](./example/DeepSeek/README.md#deepseek-v32-w4a8) 量化，单卡64G显存，100G内存即可执行。
- [2025/12/5] 🚀 msModelSlim 支持 [Qwen3-VL-235B-A22B W8A8](./example/multimodal_vlm/Qwen3-VL-MoE/README.md) 量化。
- [2025/10/16] 🚀 msModelSlim 支持 [Qwen3-235B-A22B W4A8](./example/Qwen3-MOE/README.md#qwen3-235b-a22b-w4a8-混合量化)、[Qwen3-30B-A3B W4A8](./example/Qwen3-MOE/README.md#qwen3-30b-a3b-w4a8-混合量化) 量化。vLLM Ascend已支持量化模型推理部署 [部署指导](https://vllm-ascend.readthedocs.io/en/latest/user_guide/feature_guide/quantization.html#)
- [2025/09/30] 🚀 msModelSlim 支持 [DeepSeek-V3.2-Exp W8A8](./example/DeepSeek/README.md#deepseek-v32-w8a8) 量化，单卡64G显存，100G内存即可执行
- [2025/09/18] 🚀 msModelSlim 现已解决Qwen3-235B-A22B在W8A8量化下频繁出现“游戏副本”等异常token的问题 [Qwen3-MoE 量化推荐实践](./example/Qwen3-MOE/README.md)
- [2025/09/18] 🚀 msModelSlim 支持DeepSeek R1 W4A8 per-channel 量化【Prototype】
- [2025/09/03] 🤝 msModelSlim 支持大模型量化敏感层分析
- [2025/08/30] 🌴 msModelSlim 支持Wan2.1模型一键量化
- [2025/08/25] 🌱 msModelSlim 支持大模型逐层量化

<details close>
<summary>Previous News</summary>

- [2025/08/21] 🌱 msModelSlim 支持大模型SSZ权重量化算法

</details>

> 注： **Prototype**特性未经过充分验证，可能存在不稳定和bug问题，**beta**表示非商用特性

## msModelSlim简介

msModelSlim，全称MindStudio ModelSlim，昇腾模型压缩工具。 

昇腾模型压缩工具，一个以加速为目标、压缩为技术、昇腾为根本的亲和压缩工具。包含量化和压缩等一系列推理优化技术，旨在加速大语言稠密模型、MoE模型、多模态理解模型、多模态生成模型等。

昇腾AI模型开发用户可以灵活调用Python API接口，适配算法和模型，完成精度性能调优，并支持导出不同格式模型，通过MindIE、vLLM Ascend等推理框架在昇腾AI处理器上运行。

## 安装指南

具体安装步骤请查看[安装指南](./docs/安装指南.md)。

## 快速入门

快速入门旨在帮助用户快速通过一键量化的方式完成大模型量化功能。

具体快速入门请查看[快速入门](./docs/快速入门/一键量化快速入门.md)。

## 支持矩阵

支持矩阵旨在以表格形式呈现不同功能和模型已适配场景的情况。

具体支持矩阵请查看[支持矩阵](./docs/支持矩阵/大模型支持矩阵.md)。

## 功能指南

功能指南基于msModelSlim不同架构下的功能支持情况，提供功能使用说明和接口说明。

具体功能指南请查看[功能指南](./docs/README.md#功能指南)。

## 自主量化
面向需要将自有模型接入 msModelSlim 的开发者，提供自主将模型接入msModelSlim一键量化的指导。

具体模型接入指南请查看[自主量化模型接入指南](./docs/自主量化/模型接入.md)。

## 案例集

案例集通过具体的文字说明和代码示例，以实际应用场景为基础，旨在指导用户快速熟悉特定场景下msModelSlim工具的使用，包括一些精度调优方法等，msModelSlim将持续完善案例集。

具体案例集请查看[案例集](./docs/README.md#案例集)。

## 常见问题

相关FAQ请参考链接：[FAQ](./docs/FAQ.md)

## 其他资源
- [提issue](https://gitcode.com/Ascend/msit/issues/create?type=template&title=Bug-Report|%E7%BC%BA%E9%99%B7%E5%8F%8D%E9%A6%88&template=.gitcode%252FISSUE_TEMPLATE%252Fbug-report.yml&default_branch=master&project_path_with_namespace=Ascend%252F.gitcode)
- [提新功能诉求](https://gitcode.com/Ascend/msit/issues/create?type=template&title=%E6%96%B0%E9%9C%80%E6%B1%82&template=.gitcode%252FISSUE_TEMPLATE%252Ffeature.yml&default_branch=master&project_path_with_namespace=Ascend%252F.gitcode)

## 免责声明

### 致msModelSlim使用者

1. msModelSlim工具依赖的transformers、PyTorch等第三方开源软件，均由第三方社区提供和维护，因第三方开源软件导致的问题的修复依赖相关社区的贡献和反馈。您应理解，msModelSlim仓库不保证第三方开源软件本身的问题进行修复，也不保证会测试或纠正所有第三方开源软件的漏洞和错误。
2. 在您使用msModelSlim工具时，工具通常会从硬盘中读取您从互联网所下载的模型权重（通过您提供的命令行参数或配置文件）。使用非可信的模型权重可能会导致未知的安全风险，建议您在使用工具前通过SHA256校验等方法，确保模型权重可信后再传递给工具。
3. 出于安全性及权限最小化角度考虑，您不应以root等高权限账户使用msModelSlim工具，建议您使用普通用户权限安装执行。
   - 用户须自行保证最小权限原则（如禁止 other 用户可写，常见如禁止 666、777）。
   - 使用 msModelSlim 工具请确保执行用户的 umask 值大于等于 0027，否则会导致生成的量化模型数据所在目录和权限过大。
     - 若要查看 umask 的值，可执行命令：umask
     - 若要修改 umask 的值，可执行命令：umask 新的取值
   - 请确保原始模型数据存放和量化模型数据保存在不含软链接的当前用户目录下，否则可能会引起安全问题。 

### 致数据集所有者
如果您不希望您的数据集在msModelSlim中的模型被提及，或希望更新msModelSlim中的模型关于您的数据集的描述，请在Gitcode[提issue](https://gitcode.com/Ascend/msit/issues/create?type=template&title=Bug-Report|%E7%BC%BA%E9%99%B7%E5%8F%8D%E9%A6%88&template=.gitcode%252FISSUE_TEMPLATE%252Fbug-report.yml&default_branch=master&project_path_with_namespace=Ascend%252F.gitcode)，msModelSlim将根据您的issue要求删除或更新您的数据集描述。衷心感谢您对msModelSlim的理解和贡献。

## License声明
msModelSlim提供的模型，若其模型目录中包含License文件，则遵循该文件中的许可协议。若未包含License文件，则默认适用Apache 2.0许可证。

## 致谢
msModelSlim 由华为公司的下列部门及昇腾生态合作伙伴联合贡献：

华为公司：

- 计算产品线
- 2012实验室

感谢来自社区的每一个PR，欢迎贡献 msModelSlim 。
