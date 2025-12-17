## run()

### 功能说明
运行权重压缩算法，初始化Compressor之后，通过run()函数来执行权重压缩。

### 函数原型
```python
compress_result_weight, compress_result_index, compress_result_info = compressor.run(weight_transpose=False)
```

### 参数说明
| 参数名| 输入/返回值 | 含义 | 使用限制 |
| ------ | ------ | ------ | ------ |
| weight_transpose | 输入 |待压缩权重是否需要转置。| 可选。<br>数据类型：bool。<br>默认为False，为不需要转置。可以设置为True，为需要转置。 |
| compress_result_weight | 返回值 |压缩后的权重结果。| 数据类型：dict。 |
| compress_result_index | 返回值 |压缩后的索引结果。| 数据类型：dict。 |
| compress_result_info | 返回值 |压缩信息结果。| 数据类型：dict。 |

### 调用示例
- 使用weight_path参数进行权重压缩。
```python
from msmodelslim.pytorch.weight_compression import CompressConfig, Compressor
compress_config = CompressConfig(do_pseudo_sparse=False, sparse_ratio=1, is_debug=True, compress_disable_layers=None, record_detail_root=save_root)
weight_save_path = './quant_weight.npy'  # 根据实际情况修改待压缩的权重文件路径
compressor = Compressor(compress_config, weight_path=weight_save_path)
compress_result_weight, compress_result_index, compress_result_info = compressor.run() 
```