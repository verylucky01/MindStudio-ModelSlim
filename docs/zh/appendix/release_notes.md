# 版本说明
## 版本配套说明
### 产品版本信息


| 产品名称        | 产品版本           | 版本类型 |
|-------------|----------------|------|
| msModelSlim | 26.0.0.alpha01 | 内测版本 |
| msModelSlim | 8.3.0          | 正式版本 |

### 相关产品版本配套说明

| msModelSlim版本 | CANN版本 | PyTorch版本         | torch_npu版本    | Python版本         | Transformers版本 |
|---------------|--------|-------------------|----------------|------------------|----------------|
| 26.0.0.alpha01         | 不依赖特定版本 | 与具体模型有关，请参考相关模型资料 | 与具体模型有关，请参考相关模型资料 | Python 3.10、3.11 | 与具体模型有关，请参考[example](https://gitcode.com/Ascend/msmodelslim/tree/master/example)目录下对应模型的案例说明 |
| 8.3.0         | 8.2.RC1及以上版本 | 与具体模型有关，请参考相关模型资料 | 与具体模型有关，请参考相关模型资料 | Python 3.10、3.11 | 与具体模型有关，请参考[example](https://gitcode.com/Ascend/msmodelslim/tree/master/example)目录下对应模型的案例说明 |

## 版本兼容性说明

无

## 特性变更说明

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
