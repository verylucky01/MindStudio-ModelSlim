## 项目目录结构总览

---

## 仓库顶层目录结构

```text
msmodelslim/
├── ascend_utils/          # 昇腾 Ascend 相关工具与适配（推理、剪枝、量化、蒸馏、安全等）
├── config/                # 全局配置目录（如 config.ini）
├── docs/                  # 项目文档（安装、算法说明、API 文档、案例等）
├── example/               # 各模型/任务的量化示例脚本与示例数据
├── lab_calib/             # 量化校准使用的数据集与配置（json/jsonl、图片等）
├── lab_practice/          # 各模型系列的“最佳实践”量化/推理 YAML 配置
├── msmodelslim/           # 主代码包，包含量化核心逻辑与各框架/模型适配
├── precision_tool/        # 量化精度评估工具（precision_tool、TruthfulQA 等）
├── security/              # 安全检查模块（路径安全、类型安全等）
├── test/                  # 测试用例与测试脚本（单测、系统测试、冒烟测试、fuzz 等）
├── install.sh             # 安装脚本/环境初始化
├── requirements.txt       # Python 依赖列表
├── pytest.ini             # pytest 测试配置
├── README.md              # 项目总览与使用说明
├── LICENSE                # 开源许可证
├── OWNERS                 # 代码所有者/审批配置
└── setup.py               # 打包与安装配置脚本
```

> 说明：以上只是最外层项目结构，真正的核心代码主要集中在 `msmodelslim/` 目录中，下面重点展开。

---

## `msmodelslim/` 结构（一级目录）

```text
msmodelslim/
├── __init__.py
├── app/
├── cli/
├── common/
├── core/
├── infra/
├── ir/
├── mindspore/
├── model/
├── onnx/
├── processor/
├── pytorch/
├── quant/
├── Third_Party_Open_Souce_Software_Notice
└── utils/
```

- **`__init__.py`**：包初始化逻辑（如日志、补丁加载等）。
- **`app/`**：应用层封装，对外提供“量化服务、一键量化、自动调优、模型分析”等高层应用。
- **`cli/`**：命令行入口及子命令，实现 `msmodelslim` 的 CLI 使用方式。
- **`common/`**：与具体框架无关的通用“模型处理能力”，如知识蒸馏、低秩分解、通用剪枝。
- **`core/`**：量化抽象层(QAL)、量化服务、执行器、调优策略等核心基础组件。
- **`infra/`**：与外部服务/资源对接的基础设施，例如 AISBench/vLLM 服务、YAML 管理等。
- **`ir/`**：量化中间表示（IR），定义量化类型、算子、scheme、W8A8/W4A8 等模式。
- **`mindspore/`**：MindSpore 框架的适配与量化实现。
- **`model/`**：各模型系列（DeepSeek、Qwen、Llama、Wan 等）的适配器。
- **`onnx/`**：ONNX 模型的后训练量化、SQuant 以及相关内核与工具。
- **`processor/`**：在模型前/后对权重与激活做处理的“处理器层”，如 SmoothQuant、AutoRound、稀疏化等。
- **`pytorch/`**：PyTorch 框架的适配层、PTQ/QAT 工具和大量 C++/CUDA 扩展。
- **`quant/`**：量化 session 封装，把配置、模型、量化器、处理器串联起来。
- **`utils/`**：通用工具，如日志、配置、DAG 工具、分布式、校验、安全等。

---

## `app/` 目录结构

```text
app/                          # “应用层”封装，对外提供高层应用能力
├── __init__.py
├── analysis/                 # 模型/层级分析相关应用
│   ├── __init__.py
│   └── application.py        # 分析应用入口（封装分析流程）
├── auto_tuning/              # 自动调优应用（策略搜索、实践管理）
│   ├── __init__.py
│   ├── application.py        # 自动调优主应用入口
│   ├── evaluation_service_infra.py  # 与评估服务对接的基础设施封装
│   ├── model_info_interface.py      # 模型信息查询接口抽象
│   ├── plan_manager_infra.py        # 调优计划（Plan）管理接口
│   ├── practice_history_infra.py    # 实践历史记录管理（用于回溯和复用）
│   └── practice_manager_infra.py    # 实践（Practice）管理抽象
└── naive_quantization/       # 一键量化应用
    ├── __init__.py
    ├── application.py        # 一键量化主应用入口
    ├── model_info_interface.py      # 量化前模型信息获取接口
    └── practice_manager_infra.py    # 量化实践管理（简单版）
```

---

## `cli/` 目录结构

```text
cli/                          # 命令行入口与子命令
├── __init__.py
├── __main__.py               # `python -m msmodelslim.cli` 总入口
├── analysis/                 # 分析相关 CLI 子命令
│   ├── __init__.py
│   └── __main__.py           # 分析命令入口
├── auto_tuning/              # 自动调优相关 CLI 子命令
│   ├── __init__.py
│   └── __main__.py           # 自动调优命令入口
├── naive_quantization/       # 一键量化相关 CLI 子命令
│   ├── __init__.py
│   └── __main__.py           # 朴素量化命令入口
└── utils.py                  # CLI 公共工具函数（参数解析、输出等）
```

---

## `common/` 目录结构

```text
common/                       # 与框架无关的通用“模型处理”能力
├── __init__.py
├── knowledge_distill/        # 知识蒸馏公共实现
│   ├── __init__.py
│   └── knowledge_distill.py  # 通用蒸馏流程与接口
├── low_rank_decompose/       # 低秩分解相关能力（LoRA/低秩近似等）
│   ├── __init__.py
│   ├── export.py             # 导出低秩分解结果
│   ├── low_rank_decompose.py # 核心低秩分解逻辑
│   ├── tucker.py             # Tucker 分解实现
│   └── vbmf.py               # VBMF（变分贝叶斯矩阵分解）
└── prune/                    # 通用剪枝实现（与具体框架解耦）
    ├── __init__.py
    └── transformer_prune/
        ├── __init__.py
        └── prune_model.py    # Transformer 结构的通用剪枝逻辑
```

---

## `core/` 目录结构

```text
core/                         # 量化核心抽象层(QAL)、执行器、服务等
├── __init__.py
├── analysis_service/         # 分析服务（与 app.analysis 配合）
├── base/                     # 核心协议与处理器抽象
├── const.py                  # 核心常量定义
├── graph/                    # “图”相关抽象
├── observer/                 # 量化统计模块（Observer）
├── practice/                 # 调优/实践相关抽象
├── quant_service/            # “量化服务”核心实现（v0/v1、多模态等）
├── quantizer/                # 量化器（如何根据统计信息生成量化参数）
├── runner/                   # 执行器（驱动量化/分析流程）
└── tune_strategy/            # 调优策略（如何选配置、扫参等）
```

> 详细文件名见仓库实际结构，这里只归纳每个子模块的大致职责：

- **analysis_service/**：封装各种分析方法（如层敏感度分析）、层选择策略及其流水线接口。
- **base/**：定义 Processor/Protocol 等核心接口与基类。
- **graph/**：定义图适配器类型，用于统一不同框架/表示下的“计算图”。
- **observer/**：提供 MinMax、Histogram、RecallWindow 等多种统计方式。
- **quant_service/**：统一的量化服务框架（modelslim_v0/v1、多模态 SD/VLM），含保存到 AscendV1/MindIE 等格式的能力。
- **quantizer/**：MinMax/Histogram/SSZ/GPTQ 等量化器实现，用于把统计信息转换为量化参数。
- **runner/**：多种执行模式（层级、流水线并行、数据并行）驱动量化/分析流程。
- **tune_strategy/**：调优策略及其插件机制，例如 StandingHigh 策略。

---

## `infra/` 目录结构

```text
infra/                        # 对接外部服务/资源的基础设施层
├── __init__.py
├── aisbench_server.py        # 与 AISBench 推理服务的集成封装
├── file_dataset_loader.py    # 从文件加载数据集（统一接口）
├── service_oriented_evaluate_service.py  # 面向服务的评估服务封装
├── vllm_ascend_server.py     # vLLM + Ascend 推理服务适配
├── vlm_dataset_loader.py     # VLM 用数据集加载器
├── yaml_plan_manager.py      # 基于 YAML 的调优 plan 管理
├── yaml_practice_history_manager.py  # 实践历史（YAML 存储）管理
└── yaml_practice_manager.py  # 实践配置（YAML）管理
```

---

## `ir/`目录结构

```text
ir/                           # 量化 IR：类型、算子、配置等
├── __init__.py
├── api/                      # 对外统一 IR API（quantize/dequantize 等）
├── const.py                  # IR 层常量（如各种 QScheme）
├── qal/                      # Quantization Abstract Layer 实现（QDType/QParam/QStorage 等）
├── utils.py                  # IR 辅助工具
├── w8a8_fp_dynamic.py        # W8A8+FP8 动态量化 IR
├── w8a8_mx_dynamic.py        # W8A8 + MXFP8 动态量化 IR
└── ...                       # 其它 W4A8/W8A8 等模式的 IR 定义
```

---

## `mindspore/` 目录结构

```text
mindspore/                    # MindSpore 框架适配层
├── __init__.py
├── knowledge_distill/        # MindSpore 侧知识蒸馏
├── llm_ptq/                  # MindSpore 大模型 PTQ 支持
├── low_rank_decompose/       # MindSpore 低秩分解
├── prune/                    # MindSpore 剪枝
└── quant/                    # MindSpore 量化整体流程（PTQ）
```

- **knowledge_distill/**：MindSpore 版知识蒸馏流程。
- **llm_ptq/**：MindSpore 大模型后训练量化，含 Python 实现与多版本 `.so` 内核。
- **low_rank_decompose/**：MindSpore 下的低秩分解实现。
- **prune/**：Transformer 剪枝的 MindSpore 实现。
- **quant/ptq_quant/**：创建配置、插入量化节点、回滚、保存模型的一整套 PTQ 流程。

---

## `model/` 目录结构

```text
model/                        # 各模型系列适配器（对接具体大模型）
├── __init__.py
├── base.py                   # 模型适配基类
├── common/                   # 通用模型逻辑
├── deepseek_v3/              # DeepSeek V3 系列适配
├── deepseek_v3_2/            # DeepSeek V3.2 系列适配
├── default/                  # 默认模型适配（兜底实现）
├── hunyuan_video/            # HunYuan 视频模型适配
├── kimi_k2/                  # Kimi K2 模型适配
├── qwen1_5/                  # Qwen1.5 适配
├── qwen2/                    # Qwen2 适配
├── qwen2_5/                  # Qwen2.5 适配
├── qwen3/                    # Qwen3 适配
├── qwen3_moe/                # Qwen3-MoE 适配
├── qwen3_next/               # Qwen3-Next 适配
├── qwen3_vl/                 # Qwen3-VL 适配
├── qwen3_vl_moe/             # Qwen3-VL-MoE 适配
├── qwq/                      # QwQ 模型适配
├── wan2_1/                   # Wan2.1 模型适配
├── wan2_2/                   # Wan2.2 模型适配
├── interface.py              # 适配器统一接口
└── interface_hub.py          # 适配器注册/路由中心
```

- **common/**：封装通用的前向调用方式（层级/模型级）、Transformers 支持、VLM 基础类等。
- 其余子目录：针对特定模型系列实现相应的 `model_adapter`，用于桥接“上层量化/服务逻辑”和“底层真实模型实现”。

---

## `onnx/` 目录结构

```text
onnx/                         # ONNX 模型的量化/优化支持
├── __init__.py
├── post_training_quant/      # ONNX 后训练量化(PTQ)主实现
│   ├── config.py
│   ├── dag/                  # ONNX 图解析与 DAG 构建
│   ├── data_free/            # 无数据量化（Data-free）流程
│   ├── label_free/           # 无标注量化（Label-free）流程
│   ├── quantize.py           # PTQ 主入口
│   └── util.py
├── squant_ptq/               # SQuant PTQ 相关实现（含 aok 优化器等）
│   ├── aok/                  # 算子级优化器及变换
│   ├── onnx_ptq_kia/         # KIA 风格的 ONNX PTQ 内核
│   ├── onnx_quant_tools.py
│   ├── quant_config.py
│   └── quant_deploy.py
└── ...                       # 若干 .so 内核库和工具
```

---

## `processor/` 目录结构

```text
processor/                    # “处理器层”，在模型前后对参数/激活做处理
├── __init__.py
├── anti_outlier/             # 各类 SmoothQuant / FlexSmooth / IterSmooth 异常值抑制
├── base.py                   # Processor 基类
├── common/                   # 通用模块级处理函数
├── container/                # 处理器组合/分组
├── kv_smooth/                # KV Cache 相关平滑与监听
├── memory/                   # 中间结果加载
├── quant/                    # 与量化强相关的 Processor（AutoRound/FA3 等）
├── quarot/                   # QuaRot 相关处理器
└── sparse/                   # 稀疏化处理（ADMM/FloatSparse 等）
```

- **anti_outlier/**：异常值抑制系列算法的实现及公共组件。
- **quant/**：AutoRound、FA3 等直接参与量化精调的 Processor。
- **quarot/**：基于 Hadamard 变换的 QuaRot 算法处理器及工具。

---

## `pytorch/` 目录结构

```text
pytorch/                      # PyTorch 框架适配与扩展
├── __init__.py
├── knowledge_distill/        # PyTorch 知识蒸馏
├── llm_ptq/                  # 大模型 PTQ（PyTorch 版）
├── llm_sparsequant/          # 大模型稀疏量化
├── low_rank_decompose/       # PyTorch 低秩分解
├── lowbit/                   # 低比特算子内核
├── mindspeed_adapter/        # 与 MindSpeed 等系统适配
├── multi_modal/              # 多模态相关加速/缓存
├── prune/                    # PyTorch 剪枝
├── quant/                    # PTQ/QAT 工具（含大量 C++/CUDA 内核）
├── ra_compression/           # Rope/Attention 相关压缩
├── sparse/                   # 稀疏训练/推理支持
└── weight_compression/       # 权重量化/压缩与安全相关
```

> 该目录下有大量 `.so` 扩展库和工具脚本，是 PyTorch 相关能力的主要实现区域。

---

## `quant/` 目录结构

```text
quant/                        # 量化“会话”层封装
├── __init__.py
└── session/
    ├── __init__.py
    └── session.py            # Session 抽象：把配置、模型、processor、quantizer 串起来
```

---

## `utils/` 目录结构

```text
utils/                        # 各种通用工具函数和基础设施
├── __init__.py
├── cache/                    # 内存/PyTorch 模型缓存
├── config_map.py             # 配置映射工具
├── config.py                 # 通用配置加载/管理
├── dag_utils/                # DAG/模型结构相关工具与 Hook
├── dependency_check.py       # 依赖检查工具
├── distributed/              # 分布式相关辅助（初始化、ops、helper 等）
├── exception_decorator.py    # 统一异常装饰器
├── exception.py              # 自定义异常类型
├── function_hijacker.py      # 函数劫持工具（注入 Hook）
├── graph_utils.py            # 图相关通用工具
├── hook_utils.py             # Hook 管理工具
├── logging.py                # 日志系统封装
├── memory.py                 # 内存使用工具
├── patch/                    # 对外部库(pydantic/torch)的补丁
├── plugin/                   # 插件系统（插件加载、类型化配置与工厂）
├── security/                 # 安全相关检查（模型、路径、请求、shell 命令等）
├── seed.py                   # 统一随机种子工具
├── validation/               # 类型/值/转换校验工具
└── yaml_database.py          # 基于 YAML 的轻量“数据库”封装
```

---

如果后续项目结构有调整，可以以此文件为基础进行增删修改，以保持文档与代码目录的一致性。

