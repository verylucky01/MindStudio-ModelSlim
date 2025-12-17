## QatConfig

### 功能说明 
量化参数配置类，保存量化过程中配置的参数。

### 函数原型
```python
QatConfig(w_bit=8, a_bit=8, a_sym=False, amp_num=0, steps=1, ema=0.99, is_forward=False, ignore_head_tail_node=False, disable_names=None, has_init_quant=False, quant_mode=True, grad_scale=0.0, compressed_model_checkpoint=None, opset_version=11, save_params=False, input_names=None, output_names=None, save_onnx_name=None)
```

### 参数说明
| 参数名     | 输入/返回值 | 含义                                                       | 使用限制                                            |
|---------| ------ |----------------------------------------------------------|-------------------------------------------------|
| w_bit   | 输入 | 权重量化bit。                                                 | 可选。<br> 数据类型：int。<br>默认为8，不支持修改。                |
| a_bit   | 输入 | 激活层量化bit。                                                | 可选。<br>数据类型：int。<br>默认为8，不支持修改。                 |
| a_sym   | 输入 | 激活值是否对称量化。                                               | 可选。<br>数据类型：bool。<br>默认为False。                  |
| amp_num | 输入 | 自动回退层数。<br>精度降低过多时，可增加回退层数，推荐优先回退1~3层，如果精度恢复不明显，再增加回退层数。 | 可选。<br>数据类型：int。<br>取值范围为[0,10]，默认为0，可输入1、2、3等。 |
| steps   | 输入 | 自动回退的步数。| 可选。<br>数据类型：int。<br>默认为1，取值范围大于等于1。             |
| ema   | 输入 | Adam优化器中参数，指数移动平均数指标。 | 可选。<br> 数据类型：float。<br>取值范围为[0.1,1.0]，默认为0.99。  |
| is_forward   | 输入 |是否参考mmdetection对前向进行处理。 | 可选。<br>数据类型：bool。<br>默认为False。                  |
|ignore_head_tail_node|输入|是否将首尾层忽略，不进行量化。| 可选。<br>数据类型：bool。<br>默认为False。                  |
|disable_names|输入|需排除量化的节点名称，即手动回退的量化层名称。<br> 如精度太差，可以选择回退的量化层。| 可选。<br>数据类型：list[str]。<br>默认为None。              |
|has_init_quant|输入|模型是否做过量化初始化。| 可选。<br>数据类型：bool。<br>默认为False。                  |
|quant_mode|输入|是否开启量化模式。| 可选。<br>数据类型：bool。<br>默认值为True。                  |
|grad_scale|输入|梯度补偿力度。| 可选。<br>数据类型：float。<br>默认值为0.0，建议配置为0.001。       |
|compressed_model_checkpoint|输入|导出ONNX模型时，保存的伪量化模型权重文件及所在路径。| 可选。<br>数据类型：string。<br>默认为None。                 |
|opset_version|输入|导出ONNX模型时版本号。需提前安装对应的ONNX版本| 可选。<br>数据类型：int。<br>可选值为11和13，默认为11。      |
|save_params|输入|导出时是否将量化相关参数保存为npy文件。| 可选。<br>数据类型：bool。<br>默认为False。                  |
|input_names|输入ONNX的输入名称。| 可选。<br>数据类型：list[str]<br>默认为None。               |
|output_names|输入|ONNX的输出名称。| 可选。<br>数据类型：list[str]<br>默认为None。               |
|save_onnx_name|输入|伪量化模型权重。| 可选。<br>数据类型：str。<br>默认为None。                    |


### 调用示例
```python
from msmodelslim.pytorch.quant.qat_tools import QatConfig
quant_config = QatConfig(grad_scale=0.001)
```