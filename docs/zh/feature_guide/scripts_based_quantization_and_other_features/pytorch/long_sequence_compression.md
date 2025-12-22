# 长序列压缩
## Alibi编码类型

Alibi编码是一种位置编码方法，与RazorAttention结合使用，通过Alibi编码来识别哪些注意力头对位置信息更为敏感，从而决定哪些头可以被压缩。Alibi编码并不直接在网络中加入显式的位置编码，而是通过在query-key注意力分数上施加一个与距离成比例的偏置实现位置信息的建模。

KV Cache的管理需考虑batch, seq_len, num_heads和head_size这四个维度，其中seq_len维度通常是压缩的重点，因为随着序列长度的增加，KV Cache的内存占用会迅速增长。传统的压缩方法可能会忽略不同注意力头（heads）之间的差异，而RazorAttention加速技术则提供了一种更细粒度的内存压缩方法，针对使用Alibi编码的模型进行优化，可以更有效地识别哪些注意力头对于位置信息更为敏感，并据此调整压缩策略。RazorAttention加速技术支持全量加速和增量加速：

全量加速：压缩后的KV Cache可直接用于模型推理，实现全量加速。

增量加速：支持只更新和压缩新token对应的KV Cache部分。

目前支持对表1中Alibi编码的大模型进行长序列压缩（包括但不限于）。

表1 已验证模型列表
|模型名称|框架|
|----|-----|
|baichuan2-13b|PyTorch|

### 前提条件
已参考[安装指南](../../../install_guide.md)完成开发环境配置。
执行命令安装如下依赖。
以下命令若使用非root用户安装，需要在安装命令后加上--user，例如：pip3 install numpy==1.26.4 --user。
```
pip3 install numpy==1.26.4
pip3 install transformers==4.43.1 
pip3 install torch==2.1.0   # 安装CPU版本的PyTorch 2.1.0（不依赖torch_npu）
```
### 功能实现流程

图1 压缩接口调用流程
![Alibi压缩接口调用流程](../figures/[pytorch]alibi_compress_api_calling.png)

关键步骤说明如下：

用户准备原始模型。
调用RACompressConfig接口生成压缩配置，新建模型的压缩脚本run.py。

执行压缩算法RACompressor启动长序列压缩任务，进行长序列压缩。

调用get_alibi_windows接口导出压缩窗口，并在指定路径中获取.pt文件，具体请参见MindIE的[“加速库支持模型列表”](https://www.hiascend.com/document/detail/zh/mindie/20RC1/modellist/mindie_modellist_0001.html)章节中已适配量化的模型。

### 压缩步骤（以baichuan2-13b为例）

用户准备原始模型。

用户需要自行准备模型、权重文件。本样例以baichuan2-13b为例，从该网站下载权重文件，并上传至服务器的“baichuan2-13b”文件夹，目录示例如下：
```
config.json
configuration_baichuan.py
cut_utils.py
generation_config.json
generation_utils.py
handler.py
model-00001-of-00003.safetensors
model-00002-of-00003.safetensors
model-00003-of-00003.safetensors
modeling_baichuan_cut.py
modeling_baichuan.py
model.safetensors.index.json
pytorch_model.bin.index.json
quantizer.py
special_tokens_map.json
tokenization_baichuan.py
tokenizer_config.json
tokenizer.model
```

新建模型的压缩脚本run.py，将如下样例代码导入run.py文件，并执行。

```python
from msmodelslim.pytorch.ra_compression import RACompressConfig, RACompressor
from transformers import AutoTokenizer, AutoModelForCausalLM
config = RACompressConfig(theta=0.00001, alpha=100)   # 压缩类的配置，需根据实际情况进行修改
input_model_path = "/data1/models/baichuan/baichuan2-13b/float_path/"    # 模型权重文件的保存路径，需根据实际情况进行修改
save_path = "./win.pt"   # 生成压缩窗口的路径，需根据实际情况进行修改
tokenizer = AutoTokenizer.from_pretrained(
    pretrained_model_name_or_path=input_model_path, 
    local_files_only=True
    ) 
model = AutoModelForCausalLM.from_pretrained(
    pretrained_model_name_or_path=input_model_path, 
    local_files_only=True
    ).float().cpu()   # 不支持使用npu方式进行加载
ra = RACompressor(model, config) 
ra.get_alibi_windows(save_path)
```
执行以下命令，启动长序列压缩任务，并在“baichuan2-13b”文件夹的路径下获取.pt文件。
python3 run.py
.pt文件可用于后续的推理部署任务，具体请参见MindIE的[“加速库支持模型列表”](https://www.hiascend.com/document/detail/zh/mindie/20RC1/modellist/mindie_modellist_0001.html)章节中已适配量化的模型。

## RoPE编码类型
RoPE（Rotary Position Embedding）编码是一种高效的位置编码方式，有以下特点：

旋转编码：通过旋转操作将位置信息编码到每个token的嵌入向量中。这种旋转操作确保了模型能够捕捉到序列中元素的相对位置信息，而不依赖于绝对位置。
维度保持：旋转操作在每个维度上独立进行，有助于模型在不同的特征维度上捕获位置信息。
计算效率：不需要额外的参数来编码位置信息，而是通过数学旋转操作来实现，计算效率较高。
通过RoPE编码与RazorAttention结合，可分析注意力头对位置编码的依赖性，来决定哪些头可以被压缩，以优化模型的存储、传输和计算效率，提高模型在实际应用中的可部署性和实用性。

利用RoPE编码的位置信息：由于RoPE编码已经有效地将位置信息编码到每个token中，RA算法可以利用这一点来更好地识别哪些注意力头对于位置信息更为敏感，从而更有针对性地进行压缩。
优化压缩策略：通过结合RoPE编码和RA算法，可以在保持模型性能的同时，针对不同的注意力头实施更精细的压缩策略。例如，对于依赖位置信息的Retrieval Head，可以保持其KV Cache的完整性，而对于不依赖位置信息的Non-retrieval Head，则进行压缩。
目前支持对表1中RoPE编码的大模型进行长序列压缩（包括但不限于）。

表1 已验证模型列表

| 模型名称             | 框架    |
|----------------------|--------|
| Qwen2-72b-instruct   | PyTorch|
| llama3.1-70b         | PyTorch|

### 前提条件
已参考[安装指南](../../../install_guide.md)完成开发环境配置。
执行命令安装如下依赖。
以下命令若使用非root用户安装，需要在安装命令后加上--user，例如：pip3 install numpy==1.26.4 --user。

> **说明**：torch_npu 的 xxx 需要根据实际情况进行选择，具体请参考[安装指南](../../../install_guide.md)中 torch_npu 版本。

```
pip3 install numpy==1.26.4
pip3 install transformers==4.43.1 
pip3 install torch==2.1.0       # 安装CPU版本的PyTorch 2.1.0（依赖torch_npu）
pip3 install torch_npu-2.1.0.xxx.whl
```

### 功能实现流程

图2 压缩接口调用流程

![RoPE压缩接口调用流程](../figures/[pytorch]rope_compress_api_calling.png)

关键步骤说明如下：

用户准备原始模型。

调用RARopeCompressConfig接口生成压缩配置，并新建模型的压缩脚本run.py。

调用RARopeCompressor启动长序列压缩任务，进行长序列压缩。

调用get_compress_heads接口导出需保留的Head信息，并在指定路径获取.pt文件。

用户可根据.pt文件进行压缩

压缩后的文件可用于后续的推理部署，具体请参见MindIE的[“加速库支持模型列表”](https://www.hiascend.com/document/detail/zh/mindie/20RC1/modellist/mindie_modellist_0001.html)章节中已适配量化的模型。

### 压缩步骤（以Qwen2-72b-instruct为例）

用户准备原始模型。

用户需要自行准备模型、权重文件。本样例以Qwen2-72b-instruct为例，从该网站下载权重文件，并上传至服务器的“Qwen2-72b-instruct”文件夹内，目录示例如下：
```
config.json
generation_config.json
merges.txt
model-00001-of-00037.safetensors
......
model-00037-of-00037.safetensors
model.safetensors.index.json
tokenizer.json
tokenizer_config.json
vocab.json
```

新建模型的量化脚本run.py，并将如下样例代码导入run.py文件，并执行以下命令。

```python
import torch
from msmodelslim.pytorch.ra_compression import RARopeCompressConfig, RARopeCompressor
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch_npu
torch.npu.set_compile_mode(jit_compile=False)
config = RARopeCompressConfig(induction_head_ratio=0.14, echo_head_ratio=0.01)
save_path = "./win.pt" 
model_path = "./Qwen2-72B-Instruct/"
 
model = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path=model_path,
        torch_dtype=torch.bfloat16, 
        device_map="auto", 
        local_files_only=True
    ).eval()
 
tokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path=model_path,
        pad_token='<|extra_0|>',
        eos_token='<|endoftext|>',
        padding_side='left',
        local_files_only=True
    ) 
ra = RARopeCompressor(model, tokenizer, config) 
ra.get_compress_heads(save_path)
```
启动长序列压缩任务，并在“Qwen2-72b-instruct”文件夹的路径下获取需要保留KV Cache的Head信息的.pt文件。

用户可根据.pt文件进行压缩。

压缩后的文件可用于后续的推理部署，具体请参见MindIE的[“MindIE支持模型列表”](https://www.hiascend.com/document/detail/zh/mindie/20RC1/modellist/mindie_modellist_0001.html)章节中已适配量化的模型。
```
python3 run.py
```
