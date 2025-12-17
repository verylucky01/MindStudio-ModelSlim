# GLM-4.1V 量化案例

## 模型介绍

- [GLM-4.1V-9B-Thinking](https://github.com/zai-org/GLM-V) 是由智谱 AI 联合清华大学团队推出的多模态大模型，该模型引入思考范式，并通过课程采样强化学习 RLCS（Reinforcement Learning with Curriculum Sampling）全面提升模型能力。


## 环境配置

- 基础环境配置请参考[安装指南](../../../docs/安装指南.md)
- 针对GLM-4.1V，transformers版本需要安装为4.53.0：
    - pip install transformers==4.53.0

## GLM-4.1V模型当前已验证的量化方法

| 模型       | 原始浮点权重 | 量化方式 | 推理框架支持情况| 量化命令 |
|------------|-------------|---------|----------------|---------|
| GLM-4.1V-9B-Thinking | [GLM-4.1V-9B-Thinking](https://huggingface.co/zai-org/GLM-4.1V-9B-Thinking/tree/main) | W8A8SC量化 | MindIE 预计3.0.RC1版本支持<br>vLLM Ascend 当前不支持 | [W8A8SC量化](#glm-4.1v-w8a8sc) |

## 生成量化权重

- 量化权重统一使用[quant_glm41v.py](./quant_glm41v.py)脚本生成，以下提供GLM-4.1V-9B-Thinking模型量化权重生成快速启动命令。

### 使用案例
- 如果需要使用NPU多卡量化，请先配置多卡环境变量（Atlas 300I Duo 系列产品不支持多卡量化）：
  ```shell
  export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
  export PYTORCH_NPU_ALLOC_CONF=expandable_segments:False
  ```
- 若加载自定义模型，调用`from_pretrained`函数时要指定`trust_remote_code=True`，让修改后的自定义代码文件能够正确地被加载。（请确保加载的自定义代码文件的安全性）
  

#### 1. GLM-4.1V系列
##### <span id="glm-4.1v-w8a8sc">1.1 GLM-4.1V-9B-Thinking W8A8SC量化 异常值抑制算法使用m2</span>
该示例在NPU上生成GLM-4.1V-9B-Thinking模型的量化权重。使用m2算法进行异常值抑制。

请将{浮点权重路径}和{W8A8S量化权重路径}替换为用户实际路径。{校准集图片路径}默认为"../calibImages"，以当前"../calibImages"目录中2张图片为例，实际量化时为保证精度需要从COCO数据集中扩充到30张图片。此外，用户可根据实际场景替换为其他图片。

Atlas 300I DUO 使用以下方法稀疏量化
- 稀疏量化
  ```shell
  python quant_glm41v.py \
    --model_path {浮点权重路径} \
    --save_directory {W8A8S量化权重路径} \
    --calib_images {校准集图片路径} \
    --w_bit 4 \
    --a_bit 8 \
    --device_type npu \
    --anti_method m2 \
    --is_lowbit True \
    --fraction 0.01 \
    --use_sigma True \
    --torch_dtype fp16 \
    --trust_remote_code True
  ```
- 权重压缩

  **注意**：权重压缩需要先安装MindIE
  ```shell
  # TP数为tensor parallel并行个数
  export IGNORE_INFER_ERROR=1
  torchrun --nproc_per_node {TP数} -m examples.convert.model_slim.sparse_compressor --model_path {W8A8S量化权重路径} --save_directory {W8A8SC量化权重路径}
  ```

### 量化参数说明
| 参数名 | 含义 | 默认值 | 使用方法 | 
| ------ | ---- | --- | -------- | 
| model_path | 浮点权重路径 | 无默认值 | 必选参数；<br>输入原始浮点权重目录路径。 |
| calib_images | 校准集图片路径 | ../calibImages | 可选参数；<br>输入校准数据集的目录路径。本示例中图片来源于公开数据集[COCO](https://cocodataset.org/#download)。 为保证量化精度需要根据示例扩充到30张图片。用户可根据实际场景替换为其他图片。|
| save_directory | 量化权重路径 | 无默认值 | 必选参数；<br>输出量化权重路径。 |
| part_file_size | 量化权重文件大小，单位是GB | 默认为None时不限制单个权重文件大小，此时只生成一个量化权重文件。 | 可选参数；<br>生成量化权重文件大小，请用户自定义单个量化权重文件的大小上限。|
| w_bit | 权重量化bit | 8 | 可选参数;<br>在GLM-4.1V-9B-Thinking量化场景下支持配置为4或8。|
| a_bit | 激活值量化bit | 8 | 可选参数;<br>在GLM-4.1V-9B-Thinking量化场景下支持配置为8。|
| device_type | 量化运行设备类型 | 'cpu' | 可选参数;<br>可选值：['cpu', 'npu']。 |
| trust_remote_code | 是否信任自定义代码 | False | 可选参数;<br>指定`trust_remote_code=True`让修改后的自定义代码文件能够正确地被加载（请确保所加载的自定义代码文件来源可靠，避免潜在的安全风险）。 |
| anti_method | 异常值抑制算法 | 'm2' | 可选参数;<br>可选值：['m2']。'm2'对应多模态理解模型场景下优化后的Outlier Suppression Plus异常值抑制算法。 |
| act_method | 激活值量化方法 | 2 | 可选参数;<br>(1) 1代表Label-Free场景的min-max量化方式。<br>(2) 2代表Label-Free场景的histogram量化方式。<br>(3) 3代表Label-Free场景的自动混合量化方式。|
| open_outlier | 是否开启权重异常值划分 | True | 可以配置为True或者False。 <br>设置为True时开启权重异常值划分，反之则关闭。|
| is_dynamic | 是否使用动态量化，即W8A8中的激活量化参数动态生成 | False | 可以配置为True或者False。 <br>设置为True时使用动态量化，反之则不使用。|
| is_lowbit | 是否使用稀疏量化的lowbit算法 | False | 可以配置为True或者False。 <br>设置为True时，表示使用稀疏量化的lowbit算法，反之则不使用。 <br>在`w4a8_dynamic per-group`量化场景下需要设置为True。|
| co_sparse	| 是否开启稀疏量化功能 | False | True: 使用稀疏量化功能；<br>False: 不使用稀疏量化功能。 |
| fraction | 模型权重稀疏量化过程中被保护的异常值占比  | 0.01 | 取值范围[0.01,0.1]。|
| do_smooth | 是否启动smooth量化功能 | False | True: 开启smooth量化功能；<br>False: 不开启smooth量化功能。 |
| use_sigma | 是否启动sigma功能 | False | True: 开启sigma功能；<br>False: 不开启sigma功能。 |
| sigma_factor | sigma功能中sigma的系数 | 3.0 | 数据类型为float，默认值为3.0，取值范围为[1.0, 3.0]。<br>说明：仅当use_sigma为True时生效。 |
| torch_dtype | 设置加载权重的数据类型 | bf16 | 可选值：['bf16', 'fp16']。默认值为bf16。 |
| group_size | per-group量化的分组数量 | 64 | 设置为64，128，256，512。 <br>在`w4a8_dynamic per-group`量化场景下仅支持256。|
| mindie_format | 多模态理解模型量化后的权重配置文件是否兼容MindIE现有版本 | False | 开启`mindie_format`时保存的量化权重格式能够兼容MindIE当前的版本，不开启`mindie_format`时保存的量化权重可用于vLLM Ascend部署。 |

- 更多参数配置要求，请参考量化过程中配置的参数 [QuantConfig](../../../docs/接口说明/Python-API接口说明/大模型压缩接口/大模型量化接口/PyTorch/QuantConfig.md)
  以及量化参数配置类 [Calibrator](../../../docs/接口说明/Python-API接口说明/大模型压缩接口/大模型量化接口/PyTorch/Calibrator.md)
