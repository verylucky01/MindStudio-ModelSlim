## ReStepSearchConfig

### 功能说明
采样优化参数配置类，保存用于采样优化搜索的配置参数。

### 函数原型
```python
class ReStepSearchConfig(videos_path=None, save_dir=None, neighbour_type='uniform', monte_carlo_iters=5, num_sampling_steps=50)
```

### 参数说明
| 参数名| 输入/返回值 | 含义 | 使用限制 |
| ------ | ------ | ------ | ------ |
| videos_path | 输入 | 原始模型生成的校准视频所在目录。| 必选。<br>数据类型：str。<br>用于存放使用原始步数生成的视频文件。|
| save_dir | 输入 | 搜索结果保存目录。| 必选。<br>数据类型：str。<br>用于保存搜索和优化后的时间步结果。|
| neighbour_type | 输入 | 采样过程中使用的邻域类型。| 可选。<br>数据类型：str。<br>可选值['uniform', 'random']，默认值为'uniform'。|
| monte_carlo_iters | 输入 | 采样过程中的视频采样数量。| 可选。<br>数据类型：int。<br>默认值为5。<br>控制在采样过程中要采样的视频数量。为了保障搜索结果质量，推荐合法输入范围为 5-20|
| num_sampling_steps | 输入 | 稳定扩散推理步数。| 可选。<br>数据类型：int。<br>默认值为50。<br>设置稳定扩散模型的推理步数。|


### 调用示例
```python
from msmodelslim.pytorch.multi_modal.sampling_optimization import ReStepSearchConfig

# set restep search config
config = ReStepSearchConfig(
    videos_path='/path/of/your/calibration_videos',
    save_dir='/path/to/save/searched_results',
    # set the number of sd infer steps
    num_sampling_steps=50,
)
```