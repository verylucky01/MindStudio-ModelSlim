## DitCacheSearchConfig

### 功能说明
DiT缓存搜索参数配置类，保存用于缓存搜索的配置参数。

### 函数原型
```python
class DitCacheSearchConfig(cache_ratio=1.3, dit_block_num=None, num_sampling_steps=None)
```

### 参数说明

| 参数名                | 输入/返回值 | 含义       | 使用限制                                                         | 使用说明                                                   |
|---------------------|--------------|------------|------------------------------------------------------------------|------------------------------------------------------------|
| `cache_ratio`       | 输入         | 加速比     | **可选**。<br>数据类型：`float`。<br>默认值：1.3 <br>取值范围：(1.0, 2.0)         | 控制缓存应用的加速比，值越大表示期望的加速效果越明显。     |
| `dit_block_num`     | 输入         | DiT块数量  | **可选**。<br>数据类型：`int`。<br>默认值：`None`                 | 通常由系统自动设置，无需手动指定。                         |
| `num_sampling_steps`| 输入         | 采样步数   | **必选**。<br>数据类型：`int`。<br>必须为正整数                   | 应与实际推理时的采样步数一致。                             |


### 调用示例
```python
from msmodelslim.pytorch.multi_modal.dit_cache import DitCacheSearchConfig

# 设置搜索配置
config = DitCacheSearchConfig(
    cache_ratio=1.3,
    num_sampling_steps=100
)
```

### 注意事项

1. cache_ratio应在合理范围内设置，过大的值可能导致质量下降
2. 搜索过程的时间复杂度与num_sampling_steps成正比
