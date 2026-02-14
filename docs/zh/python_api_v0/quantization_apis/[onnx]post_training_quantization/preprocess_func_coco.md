## preprocess_func_coco

### 功能说明
对图像进行预处理的函数，主要对COCO数据集进行预处理，会将图像文件读入内存，将其调整为指定的高度和宽度，然后将其转换为浮点数类型，并将像素值归一化到0到1之间，返回指定批次大小的图像数据。

### 函数原型
```python
preprocess_func_coco(data_path, height=320, width=320, batch_size=1)
```

### 参数说明
| 参数名| 输入/返回值 | 含义 | 使用限制 |
| ------ | ------ | ------ | ------ |
| data_path | 输入 | 待处理数据集所在路径。| 必选。<br>数据类型：String。|
| height | 输入 | 图片高度。| 可选，默认为320。<br>数据类型：int。<br>取值范围：大于0的整数。|
| width | 输入 | 图片宽度。| 可选，默认为320。<br>数据类型：int。<br>取值范围：大于0的整数。|
| batch_size | 输入 | 表示每个batch使用的图片数量。| 可选，默认为1。<br>数据类型：int。<br>取值范围：大于0的整数。|



### 调用示例
```python
from msmodelslim.onnx.post_training_quant import QuantConfig
from msmodelslim.onnx.post_training_quant.label_free.preprocess_func import preprocess_func_coco
calib_data = preprocess_func_coco("./test/")
quant_config = QuantConfig(calib_data = calib_data, amp_num = 5)
```