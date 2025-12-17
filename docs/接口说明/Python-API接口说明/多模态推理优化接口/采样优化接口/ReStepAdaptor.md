## ReStepAdaptor

### 功能说明
采样优化适配器类，用于搜索和优化稳定扩散模型的采样步骤，以提高推理效率。

### 函数原型
```python
ReStepAdaptor(pipeline, config)
```

### 参数说明
| 参数名| 输入/返回值 | 含义 | 使用限制 |
| ------ | ------ | ------ | ------ |
| pipeline | 输入 | OpenSoraPipeline实例。| 必选。<br>数据类型：OpenSoraPipeline。|
| config | 输入 | 采样优化搜索配置。| 必选。<br>数据类型：ReStepSearchConfig。<br>包含videos_path、save_dir等配置参数。|

### 主要方法

#### search()
执行采样步骤搜索过程，用于优化稳定扩散模型的采样步骤，以提高推理效率。该方法会分析模型在不同时间步的行为，找到对生成质量影响最大的关键时间步，从而在保持生成质量的同时减少计算开销。

**返回值：**
- 数据类型：list[float]
- 含义：搜索得到的优化后时间步列表。每个元素表示一个采样时间点，范围在[0,1]之间。

### 调用示例
```python
from msmodelslim.pytorch.multi_modal.sampling_optimization import ReStepSearchConfig, ReStepAdaptor

# 设置搜索配置
config = ReStepSearchConfig(
    videos_path='/path/of/your/calibration_videos',  # 原始模型生成的校准视频目录
    save_dir='/path/to/save/searched_results',       # 搜索结果保存目录
    neighbour_type='uniform',                        # 采样过程中使用的邻域类型
    monte_carlo_iters=5                              # 采样过程中的视频采样数量
)

# 创建适配器并执行搜索
restep_adaptor = ReStepAdaptor(pipeline, config)
scheduler_timestep: list = restep_adaptor.search()
```

### 使用说明

1. 首先需要使用原始采样步数生成一组校准视频，用于后续搜索优化。

2. 配置ReStepSearchConfig，指定校准视频路径和其他搜索参数。

3. 创建ReStepAdaptor实例并调用search()方法进行搜索。

4. 搜索完成后，可以使用返回的优化时间步进行推理，通常能获得更好的推理效率。

### 注意事项

1. 校准视频的质量会影响搜索结果的准确性，建议使用有代表性的视频样本。

2. 搜索过程可能需要一定时间，取决于monte_carlo_iters和视频数量。

3. 优化后的时间步应用于推理时，需要确保模型和数据处理流程与搜索时保持一致。