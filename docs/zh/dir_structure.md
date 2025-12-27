# 项目目录结构

```
msmodelslim/
├── ascend_utils/                    # Ascend工具集，提供昇腾AI处理器相关的工具和接口
│   ├── __init__.py
│   ├── common/                      # 通用工具模块，包含昇腾相关的通用功能
│   │   ├── __init__.py
│   │   ├── acl_inference.py          # ACL推理接口封装
│   │   ├── hook.py                  # Hook机制实现
│   │   ├── knowledge_distill/       # 知识蒸馏相关工具
│   │   │   └── [2 files: 2 *.py]
│   │   ├── mindspore_utils.py        # MindSpore工具函数
│   │   ├── prune/                   # 模型剪枝工具
│   │   │   └── [5 files: 5 *.py]
│   │   ├── security/                # 安全相关工具（路径、类型检查等）
│   │   │   └── [5 files: 5 *.py]
│   │   └── utils.py                 # 通用工具函数
│   ├── core/                        # 核心功能模块
│   │   ├── __init__.py
│   │   └── dag/                     # 有向无环图（DAG）相关功能
│   │       └── [5 files: 5 *.py]
│   ├── mindspore/                   # MindSpore框架支持
│   │   ├── __init__.py
│   │   ├── dag/                     # MindSpore DAG适配
│   │   │   └── [2 files: 2 *.py]
│   │   ├── knowledge_distill/       # MindSpore知识蒸馏
│   │   │   └── [4 files: 4 *.py]
│   │   ├── prune/                   # MindSpore模型剪枝
│   │   │   └── [1 file: 1 *.py]
│   │   └── quant/                    # MindSpore量化支持
│   │       └── [15 files: 15 *.py]
│   └── pytorch/                     # PyTorch框架支持
│       ├── __init__.py
│       ├── dag/                     # PyTorch DAG适配
│       │   └── [2 files: 2 *.py]
│       ├── knowledge_distill/       # PyTorch知识蒸馏
│       │   └── [4 files: 4 *.py]
│       └── prune/                   # PyTorch模型剪枝
│           └── [1 file: 1 *.py]
├── config/                          # 项目配置文件目录
│   ├── __init__.py
│   └── config.ini                   # 主配置文件
├── docs/                            # 项目文档目录
│   └── zh/                          # 中文文档
│       ├── algorithms_instruction/   # 算法说明文档，介绍各种量化算法原理
│       │   └── [16 files: 14 *.md, 1 *.png, 1 *.puml]
│       ├── case_studies/            # 案例研究文档，实际应用场景案例
│       │   └── [8 files: 8 *.md]
│       ├── custom_quantization/     # 自主量化指南
│       │   └── [2 files: 2 *.md]
│       ├── FAQ.md                   # 常见问题解答
│       ├── feature_guide/           # 功能指南，各功能模块使用说明
│       │   └── [41 files: 28 *.md, 8 *.png, 3 *.yaml, ...]
│       ├── foundation_model_support_matrix.md  # 基础模型支持矩阵
│       ├── install_guide.md         # 安装指南
│       ├── LICENSE                  # 文档许可证
│       ├── python_api/              # Python API文档，详细的API接口说明
│       │   └── [87 files: 87 *.md]
│       ├── quick_quantization_quick_start.md  # 快速入门指南
│       ├── README.md                # 文档目录说明
│       ├── release_notes.md         # 版本发布说明
│       └── security_statement/      # 安全声明文档
│           └── [3 files: 1 *.md, 1 *.png, 1 *.xlsx]
├── example/                         # 示例代码目录，包含各种模型的量化示例
│   ├── common/                      # 通用示例数据和工具
│   │   ├── boolq.jsonl              # BoolQ数据集示例
│   │   ├── ceval.jsonl              # C-Eval数据集示例
│   │   ├── cn_en.jsonl              # 中英翻译数据集
│   │   ├── data_list_1.jsonl       # 数据列表文件1
│   │   ├── data_list_2.jsonl       # 数据列表文件2
│   │   ├── deepseek_anti_prompt_50_v3_1.json  # DeepSeek反提示数据
│   │   ├── deepseek_anti_prompt_50.json
│   │   ├── deepseek_anti_prompt_fc.json
│   │   ├── deepseek_anti_prompt.json
│   │   ├── deepseek_calib_prompt_0528.json    # DeepSeek校准提示数据
│   │   ├── deepseek_calib_prompt_50_v3_1.json
│   │   ├── deepseek_calib_prompt_50.json
│   │   ├── deepseek_calib_prompt.json
│   │   ├── humaneval_x.jsonl       # HumanEval-X代码评估数据集
│   │   ├── llama_anti_prompt.json   # LLaMA反提示数据
│   │   ├── llama_anti_prompt.jsonl
│   │   ├── llama_calib_prompt.jsonl # LLaMA校准提示数据
│   │   ├── mix_calib.jsonl          # 混合校准数据
│   │   ├── mix_dataset_glm.json     # GLM混合数据集配置
│   │   ├── qwen_anti_prompt_72b_pdmix.json  # Qwen反提示数据
│   │   ├── qwen_anti_prompt.jsonl
│   │   ├── qwen_calib_prompt_72b_pdmix.json  # Qwen校准提示数据
│   │   ├── qwen_calib_prompt.jsonl
│   │   ├── qwen_mix_dataset.json    # Qwen混合数据集配置
│   │   ├── qwen_qwen3_cot_w4a4.json # Qwen3 CoT W4A4配置
│   │   ├── qwen3-moe_anti_prompt_50.json  # Qwen3-MoE反提示数据
│   │   ├── qwen3-moe_calib_prompt_50.json  # Qwen3-MoE校准提示数据
│   │   ├── rot_utils/               # QuaRot工具相关文件
│   │   │   └── [14 files: 11 *.txt, 3 *.py]
│   │   ├── security/                # 安全相关示例代码
│   │   │   └── [3 files: 3 *.py]
│   │   ├── teacher_qualification.jsonl  # 教师模型资格数据
│   │   ├── THIRD_PARTY_LICENSES.md  # 第三方许可证声明
│   │   ├── utils.py                 # 通用工具函数
│   │   ├── vlm_utils.py             # 视觉语言模型工具函数
│   │   └── wiki.jsonl               # Wiki数据集
│   ├── DeepSeek/                    # DeepSeek系列模型量化示例
│   │   ├── add_safetensors.py       # 添加safetensors格式支持
│   │   ├── convert_fp8_to_bf16.py   # FP8到BF16转换工具
│   │   ├── DeepSeek-R1-Distill/     # DeepSeek R1蒸馏相关
│   │   │   └── [1 file: 1 *.md]
│   │   ├── img_1.png                # 示例图片
│   │   ├── img_2.png
│   │   ├── img.png
│   │   ├── mtp_quant_module.py      # MTP量化模块
│   │   ├── quant_deepseek_w4a8.py   # DeepSeek W4A8量化示例
│   │   ├── quant_deepseek_w8a8.py   # DeepSeek W8A8量化示例
│   │   ├── quant_deepseek.py        # DeepSeek通用量化示例
│   │   └── README.md                # DeepSeek示例说明
│   ├── GLM/                         # GLM系列模型量化示例
│   │   ├── quant_glm.py             # GLM量化脚本
│   │   └── README.md
│   ├── GPT-NeoX/                    # GPT-NeoX系列模型量化示例
│   │   ├── quant_gpt_neox.py        # GPT-NeoX量化脚本
│   │   └── README.md
│   ├── HunYuan/                     # HunYuan系列模型量化示例
│   │   ├── quant_hunyuan.py         # HunYuan量化脚本
│   │   └── README.md
│   ├── InternLM2/                   # InternLM2系列模型量化示例
│   │   └── [2 files: 1 *.md, 1 *.py]
│   ├── Llama/                       # LLaMA系列模型量化示例
│   │   ├── quant_llama.py           # LLaMA量化脚本
│   │   └── README.md
│   ├── ms_to_vllm.py                # msModelSlim到vLLM格式转换工具
│   ├── multimodal_sd/               # 多模态生成模型（Stable Diffusion等）量化示例
│   │   └── [15 files: 6 *.md, 5 *.py, 4 *.txt]
│   ├── multimodal_vlm/               # 多模态理解模型（VLM）量化示例
│   │   └── [17 files: 8 *.md, 7 *.py, 2 *.jpg]
│   ├── osp1_2/                      # OpenSora Plan 1.2相关示例
│   │   └── [11 files: 5 *.py, 5 *.sh, 1 *.md]
│   ├── Qwen/                        # Qwen系列模型量化示例
│   │   └── [5 files: 3 *.py, 1 *.md, 1 *.yaml]
│   ├── Qwen3-MOE/                   # Qwen3-MoE系列模型量化示例
│   │   └── [2 files: 1 *.md, 1 *.py]
│   ├── Qwen3-Next/                  # Qwen3-Next系列模型量化示例
│   │   └── [1 file: 1 *.md]
│   └── README.md                     # 示例目录说明
├── lab_calib/                       # 实验室校准数据集目录，用于模型量化校准
│   ├── __init__.py
│   ├── anti_prompt.json             # 反提示数据（用于防止生成特定内容）
│   ├── boolq.jsonl                  # BoolQ数据集
│   ├── calib_prompt.jsonl           # 校准提示数据
│   ├── calibImages/                 # 校准图片数据
│   │   └── [2 files: 2 *.jpg]
│   ├── cn_en.jsonl                  # 中英翻译数据集
│   ├── data_list_1.jsonl            # 数据列表1
│   ├── data_list_2.jsonl            # 数据列表2
│   ├── humaneval_x.jsonl            # HumanEval-X代码评估数据集
│   ├── long_calib.jsonl             # 长文本校准数据
│   ├── mix_calib.jsonl              # 混合校准数据
│   ├── qwen2.5_w8a8_pdmix_anti_prompt.jsonl  # Qwen2.5 W8A8 PDMix反提示数据
│   ├── qwen2.5_w8a8_pdmix_calib_prompt.jsonl # Qwen2.5 W8A8 PDMix校准提示数据
│   ├── qwen3_cot_w4a4.json         # Qwen3 CoT W4A4配置
│   ├── qwen3_cot.json               # Qwen3 CoT配置
│   ├── teacher_qualification.jsonl  # 教师模型资格数据
│   ├── test.json                    # 测试数据
│   └── THIRD_PARTY_LICENSES.md     # 第三方许可证声明
├── lab_practice/                    # 实验室最佳实践配置目录，包含各模型的推荐配置
│   ├── __init__.py
│   ├── deepseek_v3/                # DeepSeek V3最佳实践配置
│   │   └── [3 files: 3 *.yaml]
│   ├── deepseek_v3_2/              # DeepSeek V3.2最佳实践配置
│   │   └── [2 files: 2 *.yaml]
│   ├── deepseek-r1-distill/        # DeepSeek R1蒸馏最佳实践配置
│   │   └── [4 files: 4 *.yaml]
│   ├── default/                    # 默认最佳实践配置
│   │   └── [1 file: 1 *.yaml]
│   ├── hunyuan_video/              # HunYuan Video最佳实践配置
│   │   └── [2 files: 2 *.yaml]
│   ├── qwen2/                      # Qwen2最佳实践配置
│   │   └── [1 file: 1 *.yaml]
│   ├── qwen2_5/                    # Qwen2.5最佳实践配置
│   │   └── [4 files: 4 *.yaml]
│   ├── qwen3/                      # Qwen3最佳实践配置
│   │   └── [8 files: 8 *.yaml]
│   ├── qwen3_moe/                  # Qwen3-MoE最佳实践配置
│   │   └── [2 files: 2 *.yaml]
│   ├── qwen3_next/                 # Qwen3-Next最佳实践配置
│   │   └── [1 file: 1 *.yaml]
│   ├── qwen3_vl_moe/               # Qwen3-VL-MoE最佳实践配置
│   │   └── [1 file: 1 *.yaml]
│   ├── qwq/                        # QwQ最佳实践配置
│   │   └── [2 files: 2 *.yaml]
│   ├── wan2_1/                     # Wan2.1最佳实践配置
│   │   └── [1 file: 1 *.yaml]
│   └── wan2_2/                     # Wan2.2最佳实践配置
│       └── [4 files: 4 *.yaml]
├── modelslim/                      # 旧包名兼容模块（向后兼容）
│   └── __init__.py
├── msitmodelslim/                  # 中间包名兼容模块（向后兼容）
│   └── __init__.py
├── msmodelslim/                    # 主代码包，包含所有核心功能模块
│   ├── __init__.py                 # 包初始化文件，设置日志和补丁
│   ├── app/                        # 应用程序层，提供高级应用接口
│   │   └── [56 files: 56 *.py]    # 包含：量化服务、自动调优、层分析、一键量化等应用
│   │                               # - quant_service: 量化服务实现（v0/v1版本、多模态支持）
│   │                               # - auto_tuning: 自动调优应用，支持策略搜索和评估
│   │                               # - analysis: 层敏感度分析应用
│   │                               # - naive_quantization: 一键量化应用
│   │                               # - tune_strategy: 调优策略接口
│   ├── cli/                        # 命令行接口模块，提供命令行工具
│   │   └── [6 files: 6 *.py]       # 包含：分析命令、量化命令等CLI入口
│   ├── common/                     # 通用模块，共享的通用功能
│   │   └── [11 files: 11 *.py]     # 通用工具和辅助函数
│   ├── core/                       # 核心模块，量化抽象层和基础组件
│   │   └── [28 files: 28 *.py]     # 包含：
│   │                               # - QAL: 量化抽象层（Quantization Abstract Layer）
│   │                               #   * 定义量化基础类型（QDType/QParam/QStorage/QScheme等）
│   │                               #   * 提供动态函数分发机制（QFuncRegistry/QABCRegistry）
│   │                               # - KIA: 内核接口抽象（Kernel Interface Abstraction）
│   │                               #   * 提供量化计算内核接口（int_quantization/fp_quantization）
│   │                               # - runner: 执行器模块（模型级/层级/流水线并行执行器）
│   │                               # - graph: 图适配器类型定义
│   │                               # - api: 基础API接口（quantize/dequantize/fake_quantize等）
│   ├── infra/                      # 基础设施模块，提供底层支撑服务
│   │   └── [4 files: 4 *.py]       # 包含：数据集加载器、实践管理器、VLM数据集加载器等
│   ├── mindspore/                  # MindSpore框架适配层
│   │   └── [28 files: 18 *.py, 10 *.so]  # MindSpore相关的量化实现和扩展库
│   ├── model/                      # 模型适配层，提供各种模型的适配器
│   │   └── [43 files: 43 *.py]    # 包含各模型系列的适配器：
│   │                               # - default: 默认模型适配器
│   │                               # - deepseek_v3/deepseek_v3_2: DeepSeek系列适配器
│   │                               # - qwen2/qwen2_5/qwen3/qwen3_moe/qwen3_next/qwen3_vl_moe: Qwen系列适配器
│   │                               # - hunyuan_video: HunYuan Video适配器
│   │                               # - wan2_1/wan2_2: Wan2系列适配器
│   │                               # - qwq: QwQ适配器
│   │                               # - common: 通用模型功能（层级/模型级前向、Transformers支持、VLM基础）
│   ├── onnx/                       # ONNX格式支持模块
│   │   └── [143 files: 100 *.so, 41 *.py, 2 *.txt]  # ONNX相关的Python接口和扩展库
│   ├── pytorch/                    # PyTorch框架适配层
│   │   └── [424 files: 250 *.so, 169 *.py, 2 *.cpp, ...]  # PyTorch相关的量化实现和扩展库
│   ├── quant/                      # 量化核心模块，实现各种量化算法和策略
│   │   └── [104 files: 89 *.py, 15 *.txt]  # 包含：
│   │                               # - ir: 量化中间表示（W4A8/W8A8/W8A8S等量化模式定义）
│   │                               # - quantizer: 量化器实现（MinMax/Histogram/SSZ等量化方法）
│   │                               # - processor: 量化处理器（SmoothQuant/IterSmooth/AutoRound/QuaRot/FA3等）
│   │                               # - observer: 观察器（MinMax/Histogram/RecallWindow等统计方法）
│   │                               # - session: 量化会话管理（配置处理和量化流程）
│   │                               # - anti_outlier: 异常值处理（SmoothQuant/FlexSmooth/IterSmooth）
│   │                               # - kv_smooth: KV Cache平滑处理
│   │                               # - sparse: 稀疏化处理（ADMM/FloatSparse）
│   │                               # - quarot: QuaRot量化算法（Hadamard矩阵相关）
│   ├── Third_Party_Open_Souce_Software_Notice  # 第三方开源软件声明
│   ├── tools/                      # 工具模块，提供辅助工具
│   │   └── [10 files: 6 *.py, 2 *.cpp, 2 *.h]  # 工具脚本和C++扩展
│   └── utils/                      # 工具函数模块，提供各种实用工具
│       └── [41 files: 41 *.py]     # 包含：
│                                   # - logging: 日志系统
│                                   # - config: 配置管理
│                                   # - cache: 缓存管理（内存/PyTorch）
│                                   # - dag_utils: DAG工具（Hook/模型结构处理）
│                                   # - distributed: 分布式支持
│                                   # - security: 安全工具（模型/路径/类型检查）
│                                   # - validation: 验证工具（类型/值/转换验证）
│                                   # - plugin: 插件系统
│                                   # - patch: 补丁系统（PyTorch/Pydantic）
├── precision_tool/                 # 精度评估工具，用于评估量化模型的精度
│   ├── __init__.py
│   ├── logger.py                   # 日志工具
│   ├── precision_tool.py          # 精度评估主工具
│   └── truthfulqa_eval.py         # TruthfulQA评估工具
├── security/                       # 安全模块，提供安全相关的检查和工具
│   ├── __init__.py
│   ├── path.py                     # 路径安全检查
│   └── type.py                     # 类型安全检查
├── test/                           # 测试目录，包含各种测试用例和测试工具
│   ├── cases/                      # 单元测试用例
│   │   └── [248 files: 246 *.py, 2 *.json]  # 测试用例脚本和测试数据
│   ├── fuzz/                       # 模糊测试
│   │   └── [71 files: 41 *.txt, 27 *.py, 1 *.json, ...]  # 模糊测试用例和测试数据
│   ├── resources/                  # 测试资源文件
│   │   └── [43 files: 23 *.json, 16 *.py, 3 *.yml, ...]  # 测试用的配置、数据和工具
│   ├── run_st.py                   # 系统测试运行脚本
│   ├── run_st.sh                   # 系统测试运行Shell脚本
│   ├── run_ut.sh                   # 单元测试运行Shell脚本
│   ├── smoke/                      # 冒烟测试
│   │   └── [32 files: 20 *.yaml, 12 *.py]  # 冒烟测试配置和脚本
│   ├── ST/                         # 系统测试（System Test）
│   │   └── [232 files: 81 *.sh, 74 *.py, 68 *.yml, ...]  # 系统测试脚本、配置和用例
│   ├── st_pr/                      # PR系统测试
│   │   └── [4 files: 2 *.sh, 1 *.py, 1 *.yaml]  # PR测试相关文件
│   └── testing_utils/              # 测试工具函数
│       └── [2 files: 2 *.py]       # 测试辅助工具
├── install.sh                      # 安装脚本
├── LICENSE                         # 项目许可证文件
├── OWNERS                          # 代码所有者文件（用于代码审查）
├── pytest.ini                      # pytest配置文件
├── README.md                        # 项目主README文档
├── requirements.txt                # Python依赖包列表
└── setup.py                        # Python包安装配置文件
```