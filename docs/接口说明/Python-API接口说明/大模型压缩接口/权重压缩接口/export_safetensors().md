## export_safetensors()

### 功能说明
将压缩后的权重保存为safetensors格式的文件，并生成对应的描述文件。

说明:Compressor中的选择weight、quant_model_description参数时，需要用此函数导出。

### 函数原型
```python
Compressor.export_safetensors(path, safetensors_name=None, json_name=None)
```

### 参数说明
| 参数名| 输入/返回值 | 含义 | 使用限制 |
| ------ | ------ | ------ | ------ |
| path | 输入 |压缩结果的保存路径。| 必选。<br>数据类型：String。 |
| safetensors_name | 输入 |	safetensors格式压缩权重文件的名称。| 可选。<br>数据类型：String。<br>本参数默认为None，输出文件名为quant_model_weight_w8a8sc.safetensors。 |
| json_name | 输入 |safetensors格式压缩权重json描述文件的名称。| 可选。<br>数据类型：String。<br>本参数默认为None，输出文件名为quant_model_description_w8a8sc.json。 |

### 调用示例
- 调用Compressor的run()方法进行权重压缩。
```python
from safetensors.torch import load_file
import json
# 导入权重压缩接口
from msmodelslim.pytorch.weight_compression import CompressConfig, Compressor
# 准备待压缩权重文件和相关压缩配置，请根据实际情况进行修改
weight_path = "./quant_model_weight_w8a8s.safetensors"       # 待压缩权重文件的路径
save_path = "./w8a8sc_llama2-7b"                          # 压缩后权重文件保存的路径
json_path = "./quant_model_description_w8a8s.json"          # 待压缩权重文件的描述文件的路径
# 使用CompressConfig接口，配置压缩参数，并返回配置实例
compress_config = CompressConfig(do_pseudo_sparse=False, sparse_ratio=1, is_debug=True, record_detail_root=save_path, multiprocess_num=8)
sparse_weight = load_file(weight_path)
with open(json_path, 'r') as f:
    quant_model_description = json.load(f)
#使用Compressor接口，输入加载的压缩配置和待压缩权重文件
compressor = Compressor(compress_config, weight=sparse_weight, quant_model_description=quant_model_description)
compress_weight, compress_index, compress_info = compressor.run()
#使用export_safetensors()接口，保存压缩后的结果文件
compressor.export_safetensors(path=save_path, safetensors_name=None, json_name=None)
```