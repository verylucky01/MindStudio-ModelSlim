## Calibrator

### 功能说明
量化参数配置类，通过Calibrator类封装量化算法。

### 函数原型
```python
Calibrator(model, cfg, calib_data=None, fuse_module_call_back=None)
```

### 参数说明
| 参数名| 输入/返回值 | 含义 | 使用限制 |
| ------ | ------ | ------ | ------ |
| model | 输入 | 待量化模型实例。| 必选。<br>数据类型：PyTorch模型。|
| cfg | 输入 | 已配置的QuantConfig类。| 必选。<br>数据类型：QuantConfig。|
| calib_data | 输入 |模型训练数据，可输入真实数据用于Label-Free量化，也可输入虚拟数据来实现Label-Free量化。| 可选。<br>数据类型：list[list[Torch.Tensor]] 或list[Torch.Tensor]。<br>如果不输入数据，在模型支持单个float格式输入且指定了input_shape时，会自动调用Label-Free量化流程。针对多个输入或者需要自定义输入格式的模型，用户可随机构造输入数据来实现Label-Free量化。|
| fuse_module_call_back | 输入 | BN融合用户自定义函数，在量化前会调用该回调。| 可选。<br>数据类型：function。<br>如果模型结构特殊，不是conv->bn并列结构的，需要用户传入自定义融合函数。|

### 调用示例
```python
from msmodelslim.pytorch.quant.ptq_tools import QuantConfig, Calibrator
disable_names = []
input_shape = [1, 3, 224, 224]
quant_config = QuantConfig(disable_names=disable_names, amp_num=0, input_shape=input_shape)
calib_data = []
image = cv2.imdecode(np.fromfile("./random_image.jpg", dtype=np.uint8), 1)
image = cv2.resize(image, (224, 224,), interpolation=cv2.INTER_CUBIC)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) 
image = torch.from_numpy(image).permute(2, 0, 1)/255
image = image.unsqueeze(0)
calib_data.append([image])     #传入一张随机图片数据，用于提高精度
calibrator = Calibrator(model, quant_config, calib_data=calib_data)
```