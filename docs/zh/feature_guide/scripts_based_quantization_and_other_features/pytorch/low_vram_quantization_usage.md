# 低显存量化

## 简介

在量化大模型的时候，显存受限或模型参数过多（如千亿级）时模型无法完整加载到显存中，量化报显存不足错误，可以启用低显存量化模式。  
该模式将模型大部分模块存放于内存中，仅计算时使用NPU，可以限制显存使用。

### 注意：
本文中的**显存**实际含义为**NPU片上内存**，为方便用户理解，借用**显存**的表述。
## 使用前准备

依赖版本：accelerate >= 0.28.0

## 功能介绍

<font color="red">注意：开启后量化耗时更久！！！</font>

使用transformers库的from_pretrained方法加载模型时，通过调整<font color="orange">device_map</font>和<font color="orange">
max_memory</font>参数控制模型加载时的显存和内存约束。

* <font color="orange">device_map</font>：模块设备映射，设置为auto
* <font color="orange">max_memory</font>：显存和内存限制
  * 每张NPU卡显存最大值分别设置为容量的80%，卡号使用**整数**
  * cpu内存最大值配置为总内存容量

示例如下：

```python
model = AutoModelForCausalLM.from_pretrained(
    pretrained_model_name_or_path=model_path,
    local_files_only=True,
    torch_dtype='auto',
    device_map="auto",
    max_memory={
        0: "25GiB",  # NPU0最多使用25GB显存
        1: "25GiB",  # NPU1最多使用25GB显存
        2: "25GiB",  # NPU2最多使用25GB显存
        3: "25GiB",  # NPU3最多使用25GB显存
        "cpu": "500GiB",  # 加载模型时，最多使用500GB的Host侧内存    
    }
)
```

## 使用样例

[Deepseek w8a8量化示例](../../../../../example/DeepSeek/README.md)

