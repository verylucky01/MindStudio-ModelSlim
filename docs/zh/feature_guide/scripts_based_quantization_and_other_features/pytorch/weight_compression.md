# 权重压缩基本使用流程

## 编译压缩函数
- 进入python环境下的site-packages包管理路径，以下是以/usr/local/为用户所在目录、Python版本为3.11.10为例
```
cd /usr/local/lib/python3.11/site-packages/msmodelslim/pytorch/weight_compression/compress_graph/
```
- 编译weight_compression组件 `sudo bash build.sh {CANN包安装路径}/ascend-toolkit/latest`
- 上一步编译操作会得到build文件夹，给build文件夹相关权限 `chmod -R 550 build`

## 导入工具
from msmodelslim.pytorch.weight_compression import CompressConfig, Compressor

## 设置压缩工具配置
由于压缩工具调用的压缩函数已将大部分配置参数固定，因此在工具层面无需设置很多参数。
```python
class CompressConfig:
    """ The configuration for LLM weight compression """
    def __init__(self,
        do_pseudo_sparse=False,  # whether to do pseudo sparse before compression
        sparse_ratio=1,  # percentile of non-zero values after pseudo sparse
        is_debug=False,  # print the compression ratio for each weight if is_debug is True
        compress_disable_layers=[],  # the layers in compress_disable_layers will not be compressed and directly saved in compress_output
        record_detail_root='./',  # the save path for the temporary data
        multiprocess_num=1) -> object:  # multiprocess num

save_path = "./compress"
compress_config = CompressConfig(do_pseudo_sparse=False, sparse_ratio=1, is_debug=True, record_detail_root=save_path, multiprocess_num=8)
```


## 定义压缩任务类，启动
```python
compressor = Compressor(compress_config, path_save)
compress_weight, compress_index, compress_info = compressor.run()
```
说明：
1. `compressor.run()`有一个参数`bool: weight_transpose`，默认为`False`，即是否开启权重转置。目前已知chatGLM2-6B无需开启权重转置
2. 开启多进程权重压缩模式时，需要手动设置当前环境下最大可打开文件数，可参考以下命令：
    ```bash
    #check current limit
    ulimit -n

    #raise limit to 65535
    ulimit -n 65535
    ```

## 启动之后对每个层的权重进行 转numpy->转Nz->压缩->保存
调用C编译后的压缩函数，通过文件的形式进行交互，完成权重压缩过程。

## 导出压缩处理结果
```python
compressor.export(compress_weight, weight_root)
compressor.export(compress_index, index_root)
compressor.export(compress_info, info_root, dtype=np.int64) # info数据为int64格式需要特别声明，否则默认将会保存为int8的格式
```
说明：权重压缩工具在加载输入的权重文件时，存在一定的反序列化攻击安全风险。权重压缩工具通过界面提示操作存在反序列化攻击的安全风险，在加载前用户交互确认加载的权重文件无风险后，再进行后续操作。
