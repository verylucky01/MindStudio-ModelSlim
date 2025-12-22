## preprocess_func_imagenet

### 功能说明
对图像进行预处理的函数，主要对ImageNet数据集进行预处理，将图像加载、调整大小、转换颜色空间、归一化并返回一个批次的图像数据。

### 函数原型
```python
preprocess_func_imagenet(data_path, height=224, width=224, batch_size=1)
```

### 参数说明
| 参数名| 输入/返回值 | 含义 | 使用限制 |
| ------ | ------ | ------ | ------ |
| data_path | 输入 | 待处理数据集所在路径。| 必选。<br>数据类型：String。|
| height | 输入 | 图片高度。| 可选，默认为224。<br>数据类型：int。<br>取值范围：大于0的整数。|
| width | 输入 | 图片宽度。| 可选，默认为224。<br>数据类型：int。<br>取值范围：大于0的整数。|
| batch_size | 输入 | 表示每个batch使用的图片数量。| 可选，默认为1。<br>数据类型：int。<br>取值范围：大于0的整数。|



### 调用示例
```python
from msmodelslim.onnx.post_training_quant import QuantConfig
from msmodelslim.onnx.post_training_quant.label_free.preprocess_func import preprocess_func_imagenet
calib_data = preprocess_func_imagenet("./test/")
quant_config = QuantConfig(calib_data = calib_data, amp_num = 5)
```