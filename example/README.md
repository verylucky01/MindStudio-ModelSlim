# msModelSlim 推荐实践集

msModelSlim 推荐实践集提供了各种大语言模型、多模态理解模型和多模态生成模型的量化实践案例，帮助用户快速上手模型量化功能。

## 目录结构

### 大语言模型量化案例
- **[DeepSeek](./DeepSeek/)** - DeepSeek 系列模型量化案例
- **[GLM](./GLM/)** - GLM 系列模型量化案例  
- **[GPT-NeoX](./GPT-NeoX/)** - GPT-NeoX 系列模型量化案例
- **[HunYuan](./HunYuan/)** - HunYuan 系列模型量化案例
- **[InternLM2](./InternLM2/)** - InternLM2 系列模型量化案例
- **[Llama](./Llama/)** - LLaMA 系列模型量化案例
- **[Qwen](./Qwen/)** - Qwen 系列模型量化案例
- **[Qwen3-MOE](./Qwen3-MOE/)** - Qwen3-MOE 系列模型量化案例
- **[Qwen3-Next](./Qwen3-Next/)** - Qwen3-Next 系列模型量化案例

### 多模态理解模型量化案例
- **[multimodal_vlm](./multimodal_vlm/)** - 多模态理解模型量化案例
  - LLaVA 系列模型
  - Qwen-VL 系列模型
  - InternVL2 系列模型
  - Qwen2-VL 系列模型
  - Qwen2.5-VL 系列模型
  - Qwen3-VL-MoE 系列模型
  - GLM-4.1V 系列模型

### 多模态生成模型量化案例
- **[multimodal_sd](./multimodal_sd/)** - 多模态生成模型量化案例
  - Stable Diffusion 系列模型
  - Flux 系列模型
  - HunYuanVideo 系列模型
  - OpenSoraPlanV1_2 系列模型
  - Wan2.1 系列模型

### 其他功能
- **[common](./common/)** - 通用工具和校准数据
- **[osp1_2](./osp1_2/)** - OpenSora Plan 1.2 相关功能
- **[ms_to_vllm.py](./ms_to_vllm.py)** - msModelSlim 到 vLLM 格式转换工具

## 快速开始

### 环境配置
- 环境配置请参考[安装指南](../docs/安装指南.md)
- 不同模型系列可能依赖特定的版本，请参考各模型目录下的具体说明。

### 使用多卡量化功能
**重要提醒：Atlas 300I Duo 卡仅支持单卡单芯片处理器量化。**

如需使用 NPU 多卡量化，请先配置环境变量：
```shell
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:False
```