## export_quant_safetensor()

### 功能说明
量化参数配置类，通过calibrator类封装量化算法来保存量化后的权重及相关参数。

说明：考虑到量化参数存储过程中存在反序列化风险，我们已在存储环节采取针对性措施：将量化结果文件夹权限设为750，量化权重文件权限设为400，量化权重描述文件权限设为600，以此降低风险。

### 函数原型
```python
calibrator.export_quant_safetensor(output_path, safetensors_name=None, json_name=None)
```

### 参数说明
| 参数名| 输入/返回值 | 含义 | 使用限制 |
| ------ | ------ | ------ | ------ |
| output_path | 输入 | 量化后的权重及相关参数保存路径。| 必选。<br>数据类型：string。|
| safetensors_name | 输入 | safetensors格式量化权重文件的名称。| 可选。<br>数据类型：string。|
| json_name | 输入 | safetensors格式量化权重json描述文件的名称。| 可选。<br>数据类型：string。|

safetensors格式的权重文件和json描述文件。safetensors权重文件包含浮点、量化权重，量化层使用的量化权重和未量化层使用的原始浮点权重。json描述文件包含模型的所有module，并标明该module的量化或浮点类型，例如 FLOAT、W8A8。在多模态量化模型导出参数时需要使用safetensors导出格式，用于后续推理。

### 调用示例
根据实际需求，在QuantConfig初始化中完成所有参数的配置。
```python
from msmodelslim.pytorch.llm_ptq.llm_ptq_tools import Calibrator, QuantConfig
quant_config = QuantConfig(act_method=1, quant_mode=1,device="npu")
pipe = OpenSoraPipeline12.from_pretrained("open-sora/", local_files_only=True)
pipe = compile_pipe(pipe)
model = pipe.transformer   #根据模型实际路径配置
calibrator = Calibrator(model, quant_config, calib_dataset)
calibrator.run()
calibrator.export_quant_safetensor("/output_path/")
```