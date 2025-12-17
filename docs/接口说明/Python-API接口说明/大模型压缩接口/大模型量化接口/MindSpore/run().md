## run()

### 功能说明
运行量化算法，初始化Calibrator后通过run()函数来执行量化。

### 函数原型
```python
calibrator.run()
```

### 调用示例
```python
from msmodelslim.mindspore.llm_ptq import Calibrator, QuantConfig
quant_config = QuantConfig(disable_names=["lm_head"], fraction=0.01)
model = Model()    #根据模型实际情况进行加载
calibrator = Calibrator(cfg=quant_config, model=model, model_ckpt="./model.ckpt", calib_data=dataset_calib)
calibrator.run() 
calibrator.save("./quant_model.ckpt")
```