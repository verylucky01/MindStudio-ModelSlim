## get_alibi_windows

### 功能说明 
执行RACompressor后，可在指定路径下，通过get_alibi_windows()函数生成.pt文件。

### 函数原型
```python
RACompressor.get_alibi_windows(save_path)
```

### 参数说明
| 参数名| 输入/返回值 | 含义 | 使用限制 |
| ------ | ------ | ------ | ------ |
| save_path | 输入 | 长序列压缩时，head压缩窗口参数文件的保存路径。| 必选。<br>数据类型：String。 |

### 调用示例
```python
from msmodelslim.pytorch.ra_compression import RACompressConfig, RACompressor
config = RACompressConfig(theta=0.00001, alpha=100)
model = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path="baichuan2-13b/float_path/", 
                                             local_files_only=True).float().cpu()    # 需根据模型的实际路径配置
ra = RACompressor(model, config) 
ra.get_alibi_windows(save_path)
```