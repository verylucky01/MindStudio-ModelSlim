## QuantConfig

### 功能说明
量化参数配置类，保存量化过程中配置的参数。

### 函数原型
```python
QuantConfig(w_bit=8, a_bit=8, w_signed=True, a_signed=False, w_sym=True, a_sym=False, input_shape=None, act_quant=True, act_method=0, quant_mode=0, disable_names=None, amp_num=0, keep_acc=None, sigma=25, device='cpu')
```

### 参数说明
| 参数名| 输入/返回值 | 含义 | 使用限制 |
| ------ | ------ | ------ | ------ |
| w_bit | 输入 | 权重量化bit。| 可选。<br>数据类型：int。<br>默认为8，暂不支持其他bit量化，不支持修改。|
| a_bit | 输入 | 激活层量化bit。| 可选。<br>数据类型：int。<br>默认为8，暂不支持其他bit量化，不支持修改。|
| w_signed | 输入 | 是否对权重进行符号量化。| 可选。<br>数据类型：bool。<br>默认为True。|
| a_signed | 输入 | 是否对激活值进行符号量化。| 可选。<br>数据类型：bool。<br>默认为False。<br>使用relu的CV模型建议设置为False，其他模型建议设置为True。|
| w_sym | 输入 | 权重是否对称量化。| 可选。<br>数据类型：bool。<br>默认为True。|
| a_sym | 输入 | 激活值是否对称量化。| 可选。<br>数据类型：bool。<br>默认为False。|
| input_shape | 输入 | 模型输入的shape，用于Label-Free量化构造虚拟数据。<br>(1)当前仅支持单个输入，且输入数据格式为float的模型。<br>(2)针对多个输入或者需要自定义输入格式的模型，如需使用Label-Free量化，用户可自定义构造虚拟输入数据，可以通过配置calib_data参数实现多个输入，无需指定input_shape。| 可选，当模型支持动态shape时必须指定。<br>数据类型：list [list]<br>默认值：[]|
| act_quant | 输入 | 是否对激活值进行量化。| 可选。<br>数据类型：bool。<br>默认为True。<br>暂不支持修改。|
| act_method | 输入 | 激活值量化方法。| 可选。<br>数据类型：int。<br>可选值[0,1,2]，默认为0。<br>(1) 0代表Data-Free量化方法（具体由sigma参数决定）。<br>(2)1代表Label-Free场景的min-max observer方法。Label-Free场景推荐选1。<br>(3)2代表Label-Free场景的histogram observer方法。|
| quant_mode | 输入 | 量化模式。| 可选。<br>数据类型：int。<br>可选值为[0,1]，默认为0。<br>(1)0代表Data-Free量化模式。<br>(2)1代表Label-Free量化模式。|
| disable_names | 输入 |需排除量化的节点名称，即手动回退的量化层名称。<br>如精度太差，推荐回退量化敏感层，如分类层、输入层、检测head层等。| 可选。<br>数据类型：list[str]。<br>默认值[]。|
| amp_num | 输入 |混合精度量化回退层数。<br>精度降低过多时，可增加回退层数，推荐优先回退3~7层，如果精度恢复不明显，再增加回退层数。| 可选。<br>数据类型：int。<br>默认为0。|
| keep_acc | 输入 | 精度保持策略。<br>(1)admm和round_opt是用来改善权重量化，减少权重量化误差，推荐在Label-Free模式下使用，适当改善量化效果。<br>(2)easy_quant用来改善激活量化，减少激活量化误差，推荐在Label-Free模式下使用，通常能够起到较好的改善效果。| 可选。<br>数据类型：dict。<br>包含以下三种精度保持策略：<br>(1)admm策略：数据类型[bool, int]，bool配置是否开启，int配置优化迭代次数。(2)easy_quant：数据类型[bool, int]，bool配置是否开启，int配置优化迭代次数。(3)round_opt：数据类型[bool]，bool配置是否开启。<br>输入模板为：keep_acc={'admm': [False, 1000], 'easy_quant': [False, 1000], 'round_opt': False} 。|
| sigma | 输入 |Label-Free的量化统计方法。<br>建议输入值为0或25，卷积类模型使用sigma统计效果更好，transformers类模型min-max统计效果更好。| 可选。<br>数据类型：int。<br>默认为25。<br>(1)sigma=25时，使用sigma统计方法。<br>(2)sigma=0时，使用Min-Max统计方法。|
| device | 输入 |选择模型运行的device。| 可选。<br>可选值为["cpu", "npu"]。<br>数据类型：str。<br>默认为"cpu"。备注：当前仅多模态量化场景支持"npu"且多模态量化场景只支持"npu"|
### 调用示例
```python
from msmodelslim.pytorch.quant.ptq_tools import QuantConfig
disable_names = []
input_shape = [1, 3, 224, 224]
keep_acc={'admm': [False, 1000], 'easy_quant': [False, 1000], 'round_opt': False}
quant_config = QuantConfig(disable_names=disable_names, amp_num=0, input_shape=input_shape, keep_acc=keep_acc)
```