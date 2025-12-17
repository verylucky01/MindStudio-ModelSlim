## Calibrator

### 功能说明
量化参数配置类，通过Calibrator类封装量化算法。

### 函数原型
```python
Calibrator(model, cfg: QuantConfig, calib_data=None, disable_level='L0', all_tensors=None, mix_cfg: Optional[dict] = None)
```

### 参数说明
| 参数名| 输入/返回值 | 含义 | 使用限制 |
| ------ | ------ | ------ | ------ |
| model | 输入 | 模型。| 必选。<br>数据类型：PyTorch模型。 |
| cfg | 输入 | 已配置的QuantConfig类。| 必选。<br>数据类型：QuantConfig。 |
| calib_data | 输入 | LLM大模型量化校准的数据，输入真实数据用于Label-Free量化。| 可选。<br>数据类型：object。<br>默认值为None，为Data-Free场景，Label-Free场景必须输入。<br>输入模板：\[[input1],[input2],[input3]]。|
| disable_level | 输入 | 自动回退等级，在模型精度损失大可以适当提升等级，但回退层数不可以大于模型总层数。| 可选。<br>数据类型：object。<br>配置示例如下：(1)'L0':默认值，不执行回退。(2)'L1'：回退1层。(3)'L2'：回退2层。(3)'L3'：回退3层。(4)'L4'：回退4层。(5)'L5'：回退5层。<br>以此类推。|
| all_tensors | 输入 | 用于逐层量化校准的{name:tensor}。| 可选。<br>数据类型：dict。<br>默认值为None，采用默认配置即可。|
| mix_cfg | 输入 | 混合量化配置，指定{**nn.Module**的通配符或层名：量化类型}。<br>关键：**通配符区分大小写。并且实现采用Python的标准库fnmatch，因此相关用法请参考fnmatch**| 可选。<br>数据类型：dict。<br>默认值为None。<br>配置示例如下：{"\*down\*": "w8a16", "\*": "w8a8"}<br>**量化类型（字典的值）必须是内部可识别的配置，否则会报错。**<br>当下支持w8a8,w8a16,w8a8_dynamic

#### 混合量化优先级说明
当同时使用 mix_cfg 和已有的回退机制时，不同配置的**匹配优先级**依次为**按照下列顺序，从上到下匹配。先匹配先生效，不再进行后续匹配**：
1. 回退层（rollback_names）
由 disable_names/disable_level（自动回退逻辑）共同决定。
一旦命中，则回退到浮点，不再考虑 mix_cfg。
2. mix_cfg 中对层名的显式指定
如果 mix_cfg 明确指定某个层的量化类型（层名一一对应），则优先于通配符匹配。
3. mix_cfg 中的通配符规则
如果层名与通配符匹配成功，则使用通配符对应的量化类型。
4. 默认配置（QuantConfig）
若前面都无匹配，则使用cfg参数所传入的QuantConfig对象作为默认量化类型。
##### 优先级示例
假设mix_cfg设置为：
```python
mix_cfg = {
    "model.layers.0.mlp.down_proj": "w8a16",  # 层名，匹配成功
    "model.layers.1.mlp.down_pro?": "w8a16",  # fnmatch通配符，匹配成功
    "?q_proj": "w8a8_dynamic",  # fnmatch通配符，匹配失败
    "*q_proj": "float",  # fnmatch通配符，匹配成功
    "model.layers.[012].mlp.down_proj": "w8a8_dynamic",  # fnmatch通配符，匹配成功012，但因为0、1已经被匹配，所以这里只有2。
    "model.layers.[!456789].mlp.down_proj": "w8a16",  # fnmatch通配符，匹配成功0123。同理因为0、1、2已经被匹配，所以这里只有3。
    "model.layers.4.mlp.down_proj": "w8a8_dynamic",  # fnmatch会区分大小写，所以此处未匹配成功。按照默认QuantConfig处理为w8a8
    "model.layers.5.mlp.down_proj": "w8a16"  # 会被下面的disable_names先匹配，不会实际生效
}
```
并且此外还设置了：
```python
quant_config=QuantConfig(w_bit=8, a_bit=8)

disable_level="L1"
disable_names=["model.layers.5.mlp.down_proj"]
```
那么结果会是：
- model.layers.0.mlp.down_proj预期为w8a16
- model.layers.1.mlp.down_proj预期为w8a16
- model.layers.2.mlp.down_proj预期为w8a8_dynamic
- model.layers.3.mlp.down_proj预期为w8a16
- model.layers.4.mlp.down_proj预期为w8a8
- model.layers.5.mlp.down_proj为float 
- 所有q_proj层为float
- 有些层会根据自动回退算法优先级更高地回退到float，可能包含上述预期的一些层。

### 调用示例
根据实际需求，在QuantConfig初始化中完成所有参数的配置。
```python
from msmodelslim.pytorch.llm_ptq.llm_ptq_tools import Calibrator, QuantConfig
quant_config = QuantConfig(dev_type='cpu', pr=0.5, mm_tensor=False)
model = AutoModel.from_pretrained('/chatglm2-6b', 
                                  local_files_only=True,
                                  torch_dtype=torch.float32).cpu()   #根据模型实际路径配置
calibrator = Calibrator(model, quant_config, calib_data=dataset_calib, disable_level='L0')
```

#### （可选）INT8混合量化调用示例
根据实际需求，在QuantConfig初始化中完成所有参数的配置。
```python
from msmodelslim.pytorch.llm_ptq.llm_ptq_tools import Calibrator, QuantConfig
quant_config = QuantConfig(dev_type='cpu', pr=0.5, mm_tensor=False)
model = AutoModel.from_pretrained('/chatglm2-6b', 
                                  local_files_only=True, 
                                  torch_dtype=torch.float32).cpu()   #根据模型实际路径配置
mix_cfg = {
    "*down*": "w8a16",
    "*": "w8a8"
}
calibrator = Calibrator(model, quant_config, calib_data=dataset_calib, disable_level='L0', mix_cfg=mix_cfg)
```