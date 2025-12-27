

# msModelSlim


## 最新消息
- [2025.12.10] 支持 DeepSeek-V3.2-Exp W4A8 量化，单卡64G显存，100G内存即可执行。
- [2025.12.05] 支持 Qwen3-VL-235B-A22B W8A8量化。
- [2025.10.16] 支持 Qwen3-235B-A22B W4A8、Qwen3-30B-A3B W4A8 量化。vLLM Ascend已支持量化模型推理部署 [部署指导](https://vllm-ascend.readthedocs.io/en/latest/user_guide/feature_guide/quantization.html#)
- [2025.09.30] 支持 DeepSeek-V3.2-Exp W8A8 量化，单卡64G显存，100G内存即可执行
- [2025.09.18] 现已解决Qwen3-235B-A22B在W8A8量化下频繁出现“游戏副本”等异常token的问题 Qwen3-MoE 量化推荐实践
- [2025.09.18] 支持DeepSeek R1 W4A8 per-channel 量化【Prototype】
- [2025.09.03] 支持大模型量化敏感层分析
- [2025.08.30] 支持Wan2.1模型一键量化
- [2025.08.25] 支持大模型逐层量化

<details close>
<summary>Previous News</summary>

- [2025.08.21] 支持大模型SSZ权重量化算法

</details>

> 注： **Prototype**特性未经过充分验证，可能存在不稳定和bug问题，**beta**表示非商用特性

## 简介

MindStudio ModelSlim（昇腾模型压缩工具，msModelSlim），一个以加速为目标、压缩为技术、昇腾为根本的亲和压缩工具。包含量化和压缩等一系列推理优化技术，旨在加速大语言稠密模型、MoE模型、多模态理解模型、多模态生成模型等。

昇腾AI模型开发用户可以灵活调用Python API接口，适配算法和模型，完成精度性能调优，并支持导出不同格式模型，通过MindIE、vLLM Ascend等推理框架在昇腾AI处理器上运行。

## 目录结构
关键目录如下，详细目录可以参考[目录文件](docs/zh/dir_structure.md)
```
├─config             # 配置文件
├─docs               # 文档目录
├─lab_calib          # 校准集
├─lab_practice       # 最佳实践
├─msmodelslim
│  ├─app             # 应用
│  ├─cli             # 命令行和接口
│  ├─core            # 其他量化模块和组件
│  ├─infra           # 量化基础设施
│  ├─model           # 模型适配
│  ├─quant           
│  │  ├─ir           # 量化模式 
│  │  └─processor    # 算法
│  └─utils           # 通用基础设施
└─test               # 测试目录
```

## 版本说明

msModelSlim的版本说明包含msModelSlim的软件版本配套关系和软件包下载以及每个版本的特性变更说明，具体参见[版本说明](./docs/zh/release_notes.md)。

## 安装指南

具体安装步骤请查看[安装指南](docs/zh/install_guide.md)。

## 快速入门

快速入门旨在帮助用户快速通过一键量化的方式完成大模型量化功能。

具体快速入门请查看[快速入门](docs/zh/quick_quantization_quick_start.md)。

## 支持矩阵

支持矩阵旨在以表格形式呈现不同功能和模型已适配场景的情况。

具体支持矩阵请查看[支持矩阵](docs/zh/foundation_model_support_matrix.md)。

## 功能指南

功能指南基于msModelSlim不同架构下的功能支持情况，提供功能使用说明和接口说明。

具体功能指南请查看[功能指南](./docs/zh/README.md#功能指南)。

## 自主量化
面向需要将自有模型接入 msModelSlim 的开发者，提供自主将模型接入msModelSlim一键量化的指导。

具体模型接入指南请查看[自主量化模型接入指南](docs/zh/custom_quantization/integrating_models.md)。

## 案例集

案例集通过具体的文字说明和代码示例，以实际应用场景为基础，旨在指导用户快速熟悉特定场景下msModelSlim工具的使用，包括一些精度调优方法等，msModelSlim将持续完善案例集。

具体案例集请查看[案例集](./docs/zh/README.md#案例集)。

## 常见问题

相关FAQ请参考链接：[FAQ](./docs/zh/FAQ.md)。

## 安全声明

描述msModelSlim产品的安全加固信息、公网地址信息及通信矩阵等内容。详情请参见[msModelSlim工具安全声明](docs/zh/security_statement/security_statement.md)。

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
如果您不希望您的数据集在msModelSlim中的模型被提及，或希望更新msModelSlim中的模型关于您的数据集的描述，请在Gitcode[提issue](https://gitcode.com/Ascend/msmodelslim/issues)，msModelSlim将根据您的issue要求删除或更新您的数据集描述。衷心感谢您对msModelSlim的理解和贡献。

## LICENSE
msModelSlim产品的使用许可证，具体请参见[LICENSE](LICENSE)。

msModelSlim产品docs目录下的文档适用CC-BY 4.0许可证，具体请参见[LICENSE](docs/zh/LICENSE)。

## 贡献声明

1. 提交错误报告：如果您在msModelSlim中发现了一个不存在安全问题的漏洞，请在msModelSlim仓库中的Issues中搜索，以防该漏洞已被提交，如果找不到漏洞可以创建一个新的Issues。如果发现了一个安全问题请不要将其公开，请参阅安全问题处理方式。提交错误报告时应该包含完整信息。
2. 安全问题处理：本项目中对安全问题处理的形式，请通过邮箱通知项目核心人员确认编辑。
3. 解决现有问题：通过查看仓库的Issues列表可以发现需要处理的问题信息, 可以尝试解决其中的某个问题。
4. 如何提出新功能：请使用Issues的Feature标签进行标记，我们会定期处理和确认开发。
5. 开始贡献：  
  a. Fork本项目的仓库。  
  b. Clone到本地。  
  c. 创建开发分支。  
  d. 本地测试：提交前必须通过所有单元测试，包括新增的测试用例。  
  e. 提交代码。  
  f. 新建Pull Request。  
  g. 代码检视：您需要根据评审意见修改代码，并重新提交更新。此流程可能涉及多轮迭代。  
  h. 当您的PR获得足够数量的检视者批准后，Committer会进行最终审核。  
  i. 审核和测试通过后，CI会将您的PR合并入到项目的主干分支。


## 建议与交流

欢迎大家为社区做贡献。如果有任何疑问或建议，请提交[Issues](https://gitcode.com/Ascend/msmodelslim/issues)，我们会尽快回复。感谢您的支持。

## 致谢
msModelSlim 由华为公司的下列部门及昇腾生态合作伙伴联合贡献：

华为公司：

- 计算产品线
- 2012实验室

感谢来自社区的每一个PR，欢迎贡献 msModelSlim 。
