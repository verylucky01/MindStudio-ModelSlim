## save_qsin_qat_model

### 功能说明 
量化模型保存接口，将量化后模型保存为可在昇腾硬件上进行推理的.onnx模型。

### 函数原型
```python
save_qsin_qat_model(model, save_onnx_name, dummy_input, saved_ckpt, input_names)
```

### 参数说明
| 参数名            | 输入/返回值 | 含义 | 使用限制                      |
|----------------| ------ | ------ |---------------------------|
| model          | 输入 | 待量化模型实例。| 必选。<br>数据类型：PyTorch模型。    |
| save_onnx_name |输入|量化后保存的.onnx模型名称。| 必选。<br>数据类型：str。          |
| dummy_input    |输入|量化后模型输入的shape。| 必选。<br>数据类型：torch.Tensor。 |
|saved_ckpt|输入|保存的量化权重。| 必选。<br>数据类型：str。          |
|input_names|输入|onnx的输入名称。| 必选。<br>数据类型：list[str]     |


### 调用示例
```python
from msmodelslim.pytorch.quant.qat_tools import save_qsin_qat_model
save_onnx_name='./dest.onnx'      #请根据实际情况修改文件命名和路径
dummy_input = torch.ones([batch_size, 3, 224, 224]).type(torch.float32)
saved_ckpt = './saved_ckpt.pth'     #请根据实际情况修改文件命名和路径
input_names=['input']        #请根据实际情况修改文件命名
save_qsin_qat_model(model, save_onnx_name, dummy_input, saved_ckpt, input_names)
```