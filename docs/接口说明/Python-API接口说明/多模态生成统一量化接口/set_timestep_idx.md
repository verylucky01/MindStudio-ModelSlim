## set_timestep_idx()

### 功能说明
在时间步量化中，需要针对不同时间步(timestep)采用不同的量化策略(w8a8静态或w8a8动态)，此函数用于设置当前量化进行到的时间步。

### 函数原型
```python
set_timestep_idx(t_idx: int)
```

### 参数说明
| 参数名 | 输入/返回值 | 含义 | 使用限制 |
| ------ | ---------- | ---- | -------- |
| t_idx | 输入 | 多模态生成模型推理的当前时间步。 | 必选。<br>数据类型：int。|

### 调用示例
```python
from msmodelslim.pytorch.llm_ptq.llm_ptq_tools.timestep.manager import TimestepManager

# 定义时间步序列
timesteps = [0, 10, 20, 30, 40, 50]  # 示例时间步

for step_id, t in enumerate(timesteps):
    # 设置当前时间步索引
    TimestepManager.set_timestep_idx(step_id)

    # 执行模型推理
    model_output = pipeline(
        prompt="生成一张猫的图片",
        num_inference_steps=50,
        guidance_scale=7.5
    )
```