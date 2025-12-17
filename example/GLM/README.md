
# GLM 量化案例

## 模型介绍
- [GLM](https://github.com/THUDM/GLM)是智谱AI推出的最新一代预训练模型GLM-4系列中的开源版本。在语义、数学、推理、代码和知识等多方面的数据集测评中，GLM-4-9B 及其人类偏好对齐的版本GLM-4-9B-Chat均表现出超越 Llama-3-8B的卓越性能。除了能进行多轮对话，GLM-4-9B-Chat还具备网页浏览、代码执行、自定义工具调用（Function Call）和长文本推理（支持最大128K上下文等高级功能）。本代模型增加了多语言支持，支持包括日语，韩语，德语在内的26种语言。我们还推出了支持 1M 上下文长度（约 200 万中文字符）的GLM-4-9B-Chat-1M模型和基于 GLM-4-9B的多模态模型GLM-4V-9B。GLM-4V-9B具备1120 * 1120高分辨率下的中英双语多轮对话能力，在中英文综合能力、感知推理、文字识别、图表理解等多方面多模态评测中，GLM-4V-9B表现出超越GPT-4-turbo-2024-04-09、Gemini 1.0 Pro、Qwen-VL-Max 和 Claude 3 Opus的卓越性能。

## 环境配置

- 环境配置请参考[使用说明](../../docs/安装指南.md)

## 支持的模型版本与量化策略

| 模型系列 | 模型版本 | HuggingFace链接                                                 | W8A8 | W8A16 | W4A16 | W4A4  | 稀疏量化 | KV Cache | Attention | 量化命令                                                                                                                                                                                 |
|---------|---------|---------------------------------------------------------------|-----|-------|-------|------|---------|----------|-----------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **GLM** | GLM-4-9B | [GLM-4-9B](https://huggingface.co/THUDM/glm-4-9b)      | ✅ |   |   |   |  | ✅ |   | [W8A8C8](#glm-4-9b-w8a8c8量化)                                                                                                                                                            |

**说明：**
- ✅ 表示该量化策略已通过msModelSlim官方验证，功能完整、性能稳定，建议优先采用。
- 空格表示该量化策略暂未通过msModelSlim官方验证，用户可根据实际需求进行配置尝试，但量化效果和功能稳定性无法得到官方保证。
- 点击量化命令列中的链接可跳转到对应的具体量化命令


## 量化权重生成

- 量化权重统一使用[quant_glm.py](./quant_glm.py)脚本生成，以下提供GLM模型量化权重生成快速启动命令。

#### 量化参数说明
| 参数名               | 含义                   | 默认值 | 使用方法                                                                                                                    | 
|-------------------|----------------------| --- |-------------------------------------------------------------------------------------------------------------------------| 
| model_path        | 浮点权重路径               | 无默认值 | 必选参数；<br>输入GLM权重目录路径。                                                                                              |
| save_directory    | 量化权重路径               | 无默认值 | 必选参数；<br>输出量化结果目录路径。                                                                                                    |
| part_file_size    | 生成量化权重文件大小/GB             | 无默认值 | 可选参数；<br>生成量化权重文件大小，请用户自定义单个量化权重文件的最大限制。                                                                                                    |
| calib_texts       | 量化校准数据               | 无默认值                        | 可选参数；<br>校准数据集。                                                                                                    |
| calib_file        | 量化校准数据               | teacher_qualification.jsonl | 可选参数；<br>存放校准数据的json文件。                                                                                                    |
| anti_file         | 异常值抑制校准数据            | mix_dataset_glm.json | 存放异常值抑制校准数据的json文件。                                                                                                     |
| w_bit             | 权重量化bit              | 8 | 大模型量化场景下，可配置为8或16； <br>大模型稀疏量化场景下，需配置为4。                                                                                |
| a_bit             | 激活值量化bit             | 8 | 大模型量化场景下，可配置为8或16； <br>大模型稀疏量化场景下，需配置为8。                                                                                |
| disable_names     | 手动回退的量化层名称           | 默认回退所有down_proj层 | 用户可根据精度要求手动设置，默认回退隐藏层的降维投影层。                                                                                            |
| device_type       | device类型             | cpu | 可选值：['cpu', 'npu']。                                                                                                      |
| fraction          | 模型权重稀疏量化过程中被保护的异常值占比 | 0.01 | 取值范围[0.01,0.1]。                                                                                                          |
| act_method        | 激活值量化方法              | 1 | (1) 1代表Label-Free场景的min-max量化方式。 <br>(2) 2代表Label-Free场景的histogram量化方式。 <br>(3) 3代表Label-Free场景的自动混合量化方式，LLM大模型场景下推荐使用。 |
| co_sparse         | 是否开启稀疏量化功能           | False | True: 使用稀疏量化功能；<br>False: 不使用稀疏量化功能。                                                                                    |
| anti_method       | 离群值抑制参数              | 无默认值 | 'm1': SmoothQuant算法。<br>'m2': SmoothQuant加强版算法，推荐使用。<br>'m3': AWQ算法。<br>'m4': smooth优化算法 。<br>'m5': CBQ量化算法。<br>默认为m2。  |
| disable_level     | L自动回退等级              | L0 | 配置示例如下：<br>'L0'：默认值，不执行回退。<br>'L1'：回退1层。<br>'L2'：回退2层。<br>'L3'：回退3层。<br>'L4'：回退4层。<br>'L5'：回退5层。                        |
| do_smooth         | 是否启动smooth量化功能          | False | True: 开启smooth量化功能；<br>False: 不开启smooth量化功能。                                                                                  |
| use_sigma         | 是否启动sigma功能          | False | True: 开启sigma功能；<br>False: 不开启sigma功能。                                                                                  |
| use_reduce_quant  | 权重量化是否是lccl all reduce量化 | False | 用于MindIE推理的标识。 |
| tp_size           | 模拟多卡量化时的卡数 | 1 | 数据取值范围为[1,2,4,8,16]，默认值为1，不启用模拟多卡量化。<br>设置为2、4、8、16时，对于通信层的linear会进行模拟多卡，每张卡使用不同的scale和offset进行量化。 |
| sigma_factor      | sigma功能中sigma的系数 | 3.0 | 数据类型为float，默认值为3.0，取值范围为[1.0, 3.0]。<br>说明：仅当use_sigma为True时生效。 |
| is_lowbit         | 是否开启lowbit量化功能       | False | (1) 当w_bit=4，a_bit=8时，为大模型稀疏量化场景，表示开启lowbit稀疏量化功能。<br>(2) 其他场景为大模型量化场景，会开启量化自动精度调优功能。<br>当前量化自动精度调优框架支持W8A8，W8A16量化。    |
| mm_tensor         | 是否开启mm_tensor量化功能      | True | True: 开启mm_tensor量化功能；<br>False: 不开启mm_tensor量化功能。                                                                          |
| w_sym             | 是否开启w_sym量化功能      | True | True: 开启w_sym量化功能；<br>False: 不开启w_sym量化功能。                                                                          |
| use_kvcache_quant | 是否使用kvcache量化功能      | False | True: 使用kvcache量化功能；<br>False: 不使用kvcache量化功能。                                                                          |
| use_fa_quant      | 是否使用FA3量化 | False | True: 使用FA3量化类型；<br>False: 不使用FA3量化类型。|
| fa_amp | FA3量化场景下的自动回退的layer数量 | 0 | 数据类型为int，默认值为0。数据取值范围是大于等于0，并且小于等于模型layer数量，如果超出模型的layer数量将会取模型的最大layer数量为回退层数。 |
| open_outlier | 是否开启权重异常值划分 | True | True：开启权重异常值划分。<br>False：关闭权重异常值划分。<br>说明：(1)仅在lowbit设置为True时生效。(2)per_group量化场景下，需协同设置is_lowbit为True，open_outlier为False。|
| group_size | per_group量化中group的大小 | 64 | 默认值为64，支持配置为32，64，128。<br>说明:仅适用于per_group量化场景，需协同设置is_lowbit为True，open_outlier为False。|
| is_dynamic        | 是否使用per-token动态量化功能  | False | True: 使用per-token动态量化；<br>False: 不使用per-token动态量化。                                                                      |
| input_ids_name | 指定分词结果中输入 ID 对应的键名 | input_ids | 无 |
| attention_mask_name | 指定分词结果中注意力掩码对应的键名 | attention_mask | 无 |
| tokenizer_args | 加载自定义tokenizer时传入的自定义参数 | 无 | 以字典方式传入。 |
| disable_last_linear | 是否回退最后linear层 | True | True：回退最后linear层。<br>False：不回退最后linear层。 |
| model_name | 模型名称，可选参数 | None | 用于控制异常值抑制参数。 |
| trust_remote_code | 是否信任自定义代码 | False | 指定`trust_remote_code=True`让修改后的自定义代码文件能够正确地被加载(请确保所加载的自定义代码文件来源可靠，避免潜在的安全风险)。 |
| mindie_format | 非多模态模型量化后的权重配置文件是否兼容MindIE现有版本 | False | 开启`mindie_format`时保存的量化权重格式能够兼容MindIE 2.1.RC1及之前的版本。 |


- 更多参数配置要求，请参考量化过程中配置的参数 [QuantConfig](../../docs/接口说明/Python-API接口说明/大模型压缩接口/大模型量化接口/PyTorch/QuantConfig.md)
  以及量化参数配置类 [Calibrator](../../docs/接口说明/Python-API接口说明/大模型压缩接口/大模型量化接口/PyTorch/Calibrator.md)


### 使用案例
- 请将{浮点权重路径}和{量化权重路径}替换为用户实际路径。
- 如果需要使用NPU多卡量化，请先配置环境变量，支持多卡量化，但GLM-4-9B量化仅需要单卡：
  ```shell
  export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
  export PYTORCH_NPU_ALLOC_CONF=expandable_segments:False
  ```
- 若加载自定义模型，调用`from_pretrained`函数时要指定`trust_remote_code=True`让修改后的自定义代码文件能够正确的被加载。(请确保加载的自定义代码文件的安全性)

#### GLM-4-9B模型量化
##### <span id="glm-4-9b-w8a8c8量化">GLM-4-9B W8A8C8量化</span>
- 生成GLM-4-9B模型w8a8c8量化权重，使用histogram量化方式，在NPU上进行运算
  ```shell
  python3 quant_glm.py --model_path {浮点权重路径} --save_directory {W8A8C8量化权重路径} --device_type npu --act_method 2 --disable_level L0 --w_bit 8 --a_bit 8 --use_kvcache_quant True --calib_file ../common/mix_dataset_glm.json --anti_file ../common/mix_dataset_glm.json --trust_remote_code True
  ```