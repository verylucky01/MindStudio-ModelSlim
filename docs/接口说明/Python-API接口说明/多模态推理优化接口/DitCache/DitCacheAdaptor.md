# DitCacheAdaptor: DiT模型缓存适配器

## 1. 功能说明

DiT（Diffusion Transformer）缓存适配器类，用于优化DiT模型的推理性能。通过缓存和复用中间计算结果来减少计算量，在保持生成质量的同时提升推理效率。

主要特性：
- 自动搜索最优缓存配置。
- 支持增量计算和结果复用。
- 内置质量评估机制。
- 保存和复用搜索配置。

## 2. API参考

### 2.1 类定义
```python
class DitCacheAdaptor:
    def __init__(self, 
                 pipeline,
                 config: Optional[DitCacheSearchConfig] = None,
                 dit_block_path: str = "transformer.transformer_blocks")
```

### 2.2 参数规范

| 参数名 | 类型 | 必选/可选 | 说明 | 默认值 |
|--------|------|-----------|------|--------|
| pipeline | OpenSoraPipeline | 必选 | 模型pipeline实例，需要包含transformer blocks | - |
| config | DitCacheSearchConfig | 可选 | 缓存搜索配置对象 | None |
| dit_block_path | str | 可选 | transformer blocks在pipeline中的访问路径 | "transformer.transformer_blocks" |

### 2.3 异常处理

- `ValueError`: 当传入的参数无效时抛出，包括：
  - config不是DitCacheSearchConfig类型
  - dit_block_path格式无效或为空
  - 无法访问指定路径的transformer blocks
  - transformer blocks不是nn.ModuleList类型或为空

### 2.4 方法说明

#### 2.4.1 set_timestep_idx

```python
@classmethod
def set_timestep_idx(cls, t_idx: int) -> None
```

设置当前时间步索引。**必须**在每个timestep的开始时调用。

##### 参数
- t_idx (int): 当前时间步索引

##### 异常
- `ValueError`: 如果在DiT block前向传播前未调用此方法

#### 2.4.2 search

```python
def search(self,
          run_pipeline_and_save_videos: Callable,
          prompts_num: int = 1) -> DitCacheConfig
```

执行缓存配置搜索，寻找最优的缓存策略。

##### 参数

| 参数名                   | 类型         | 含义                                           | 默认值 | 备注                                                                 |
|------------------------|--------------|------------------------------------------------|--------|----------------------------------------------------------------------|
| `run_pipeline_and_save_videos` | `Callable`   | 运行 pipeline 并返回生成视频的函数                     | 无     | 输入参数：`pipeline (OpenSoraPipeline)`<br>返回值：`List[np.ndarray]`，每个视频的 shape 为 `(num_frames, h, w, c)` |
| `prompts_num`           | `int`        | 生成视频的数量                                 | `1`    | 控制生成视频的个数                                                   |


##### 返回值

返回值是一个用于配置 DiT 缓存机制的对象 `DitCacheConfig`，包含以下字段：

| 字段名               | 类型   | 含义                                           |
|--------------------|--------|------------------------------------------------|
| `cache_step_start`     | `int` | 开始使用缓存的时间步                              |
| `cache_step_interval`  | `int` | 缓存计算的时间步间隔，每隔多少步重新计算缓存              |
| `cache_block_start`    | `int` | 开始缓存的 block 索引，`0` 表示从第一个 block 开始        |
| `cache_num_blocks`     | `int` | 要缓存的 block 数量                                 |


#### 2.4.3 `update_cache_config`

用于手动更新当前的缓存策略配置，包括缓存的起始 block、数量及时间步相关参数。调用该方法可在无需重新搜索的情况下直接应用指定缓存策略。

```python
def update_cache_config(self,
                        cache_block_start: int,
                        cache_num_blocks: int,
                        cache_step_start: int,
                        cache_step_interval: int)
```

##### 参数说明

| 参数名               | 输入/输出 | 类型   | 含义                         |
|--------------------|------------|--------|------------------------------|
| `cache_block_start`    | 输入       | `int` | 开始缓存的 block 索引（从 0 开始）     |
| `cache_num_blocks`     | 输入       | `int` | 缓存的 block 数量                   |
| `cache_step_start`     | 输入       | `int` | 从该时间步开始启用缓存               |
| `cache_step_interval`  | 输入       | `int` | 每隔多少时间步重新计算一次缓存         |

##### 使用示例

```python
from msmodelslim.pytorch.multi_modal.dit_cache import DitCacheAdaptor, DitCacheSearchConfig

# 创建 adaptor，为 DiT 模型添加缓存功能
adaptor = DitCacheAdaptor(pipeline)

# 设置缓存配置
cache_config = dict(
    cache_block_start=2,
    cache_num_blocks=4,
    cache_step_start=10,
    cache_step_interval=5
)
adaptor.update_cache_config(**cache_config)

# 使用 pipeline 执行推理，生成视频
...
```



## 3. 使用指南

### 3.1 基础示例

```python
# 1. 导入必要的类
from msmodelslim.pytorch.multi_modal.dit_cache import DitCacheAdaptor, DitCacheSearchConfig

# 2. 定义运行pipeline的函数
def run_pipeline_and_save_videos(pipeline):
    """运行pipeline并返回生成的视频列表"""
    positive_prompt = "(masterpiece), (best quality), (ultra-detailed), {}"
    
    videos = pipeline(
        positive_prompt.format("a dog running on the beach"),
        num_frames=29,
        height=480,
        width=640,
        num_inference_steps=100,
        guidance_scale=7.5
    ).images
    
    return videos

# 3. 配置和初始化缓存适配器
config = DitCacheSearchConfig(
    cache_ratio=1.3,  # 缓存加速比
    num_sampling_steps=100  # 采样步数
)
cache_adaptor = DitCacheAdaptor(pipeline, config)

# 4. 执行缓存配置搜索
searched_config = cache_adaptor.search(
    run_pipeline_and_save_videos=run_pipeline_and_save_videos,
    prompts_num=1
)
```

### 3.2 使用流程
1. 初始化DitCacheAdaptor实例。
2. 在扩散循环中，每个时间步开始时调用set_timestep_idx()。
3. 调用search()方法进行缓存配置搜索。
4. 使用返回的缓存配置进行推理。

### 3.3 注意事项

1. **必须设置timestep**
    在每个timestep开始时调用`DitCacheAdaptor.set_timestep_idx(step_id)`。通常在模型的去噪循环中进行，示例如下：
    ```python
    for step_id, t in enumerate(timesteps):
        DitCacheAdaptor.set_timestep_idx(step_id)  # 必须在每个timestep开始时调用
        model_output = pipeline(...)
    ```

1. **搜索配置**

    cache_ratio推荐设置为1.3表示期望的加速比，搜索过程包含校准视频生成和配置评估可能需要较长时间，建议在性能较好的设备上运行搜索过程。

2. **配置保存和复用**

    搜索得到的配置可以保存为JSON文件，相同场景下可以直接加载使用而无需重新搜索。

3. **使用场景**

    当前支持29\*480p和93\*720p场景，可达到约1.3倍加速同时保持生成质量，不同场景可能需要重新搜索最优配置。

4. **参数一致性**

    确保搜索和推理时使用相同的模型参数配置，优化后的缓存配置应用于推理时需要确保模型和数据处理流程与搜索时保持一致，包括但不限于采样步数、图像尺寸等。

## 4. 技术原理

### 4.1 理论基础

#### 4.1.1 基本假设
DiT缓存优化基于以下核心思想：
- 在扩散过程中，相邻时间步的transformer block输出变化是渐进的。
- 某些block的计算结果可以通过增量方式获得，无需完整重算。

#### 4.1.2 数学模型
在扩散模型中，令 $h_{t,i}$ 表示时间步 $t$ 第 $i$ 个transformer block的隐状态输出：

$$ h_{t,i} = \mathcal{F}_i(h_{t,i-1}), \quad i \in [1,N] $$

其中：
- $\mathcal{F}_i$ 表示第 $i$ 个transformer block的变换函数。
- $N$ 为transformer block总数。
- $t \in [0,T]$ 为扩散时间步，$T$ 为总时间步数。

### 4.2 增量计算

#### 4.2.1 区间差异定义
对于任意block区间 $[i,j]$，定义其输出差异为：

$$ \Delta_{t,[i:j]} = h_{t,j} - h_{t,i}, \quad 1 \leq i < j \leq N $$

这表示从第 $i$ 个block到第 $j$ 个block的累积变换效果。

#### 4.2.2 连续性假设
扩散过程中，相邻时间步的transformer block输出具有局部连续性：

$$ \|\Delta_{t,[i:j]} - \Delta_{t-1,[i:j]}\| \leq \epsilon, \quad \forall t > 0 $$

其中 $\epsilon$ 为小的正数，表示可接受的误差范围。

#### 4.2.3 增量近似
基于连续性假设，可以用前一时间步的差异近似当前时间步：

$$ h_{t,j} = h_{t,i} + \Delta_{t,[i:j]} \approx h_{t,i} + \Delta_{t-1,[i:j]} $$

### 4.3 缓存策略

#### 4.3.1 基础计算
对于起始block $(i = block\_start)$，直接计算：

$$ h_{t,i} = \mathcal{F}_i(h_{t,i-1}) $$

#### 4.3.2 增量更新机制
在缓存更新时间点 $(t \bmod interval = 0)$，计算并存储区间差异：

$$ \Delta_{t,[i:j]} = \mathcal{F}_{[i:j]}(h_{t,i}) - h_{t,i} $$

其中 $\mathcal{F}_{[i:j]}$ 表示从block $i$ 到 $j$ 的复合变换。

#### 4.3.3 增量重建过程
在缓存复用期间 $(t \bmod interval \neq 0)$，使用存储的差异重建输出：

$$ h_{t,j} = h_{t,i} + \Delta_{t-\delta t,[i:j]}, \quad \delta t = t \bmod interval $$

### 4.4 工程实现

#### 4.4.1 核心架构
缓存机制通过替换DiT block的forward方法实现：

```python
def _add_cache_to_dit_block(self, dit_blocks: nn.ModuleList):
    """为transformer block添加缓存逻辑
    
    缓存处理流程:
    1. 基础条件判断:
       - t_idx < cache_step_start: 使用原始forward
       - cache禁用: 使用原始forward
    
    2. 根据block位置处理:
       - 基础block (index = cache_block_start): 计算并缓存输入
       - 中间blocks: 返回占位符DitCacheDummy
       - 复用block (index = cache_block_start + cache_num_blocks - 1): 
         使用缓存delta重建输出
       - 其他blocks: 使用原始forward
    """
```

#### 4.4.2 关键实现细节

##### 基础块处理
```python
# 基础block处理
if _block_idx == blk_start:
    self.cache[START_HIDDEN_KEY] = hidden_states
    
    if is_step_to_store_cache:
        return orig_forward(hidden_states, *args, **kwargs)
    else:
        return DitCacheDummy()
```

##### 复用块处理
```python
# 复用block处理
elif _block_idx == blk_end:
    last_block_hidden = self.cache.pop(START_HIDDEN_KEY)
    
    if is_step_to_store_cache:
        hidden_states = orig_forward(hidden_states, *args, **kwargs)
        delta = hidden_states - last_block_hidden
        self.cache[DELTA_HIDDEN_KEY] = delta
        return hidden_states
    else:
        return self.cache[DELTA_HIDDEN_KEY] + last_block_hidden
```

#### 4.4.3 配置参数
缓存机制的关键控制参数：
- `cache_step_start`: 开始使用缓存的时间步。
- `cache_step_interval`: 缓存更新间隔。
- `cache_block_start`: 起始缓存block位置。
- `cache_num_blocks`: 缓存block数量。

#### 4.4.4 实现注意事项
1. 必须在每个时间步开始时调用`set_timestep_idx()`。
2. 确保缓存参数合理设置，避免影响生成质量。
3. 注意内存使用，及时清理不需要的缓存。
