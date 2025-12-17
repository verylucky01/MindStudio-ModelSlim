## run()

### 功能说明
运行量化算法，初始化Calibrator后通过run()函数来执行量化。

### 函数原型
```python
calibrator.run(int_infer=False)
```

### 参数说明
| 参数名| 输入/返回值 | 含义 | 使用限制 |
| ------ | ------ | ------ | ------ |
| int_infer | 输入 | 是否使用int8matmul进行伪量化计算。| 可选。<br>数据类型：bool。<br>默认值为False。<br>该参数仅适用于W8A8场景，W8A16场景下该参数无效。 |

### 调用示例
根据实际需求，在QuantConfig初始化中完成所有参数的配置。
```python
from msmodelslim.pytorch.llm_ptq.llm_ptq_tools import Calibrator, QuantConfig
quant_config = QuantConfig(dev_type='cpu', pr=0.5, mm_tensor=False)
model = AutoModel.from_pretrained('/chatglm2-6b', 
                                  local_files_only=True, 
                                  torch_dtype=torch.float32).cpu()   #根据模型实际路径配置
calibrator = Calibrator(model, quant_config, calib_data=dataset_calib, disable_level='L0')
calibrator.run(int_infer=False) 
calibrator.save(quant_weight_save_path)
```
