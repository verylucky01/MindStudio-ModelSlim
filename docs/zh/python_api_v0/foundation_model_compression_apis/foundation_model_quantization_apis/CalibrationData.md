# CalibrationData

## 功能说明
混合校准集接口，通过CalibrationData类混合指定的数据集，支持用户自定义数据集

## 函数原型
```python
CalibrationData(config_path, save_path, tokenizer=None, model=None)
```

## 参数说明
| 参数名| 输入/返回值 | 含义          | 使用限制                                           |
| ----- |--------|-------------|------------------------------------------------|
| config_path | 输入     | 数据集config路径 | 必选。<br>数据类型：string。                            |
| save_path | 输入     | 混合数据集保存文件路径 | 可选，为空则不保存到文件；<br>如选择，需为json文件。<br>数据类型：string。 |
| tokenizer | 输入     | 分词器实例       | 可选。<br>数据类型：根据模型生成的PreTrainedTokenizer的子类      |
| model | 输入     | 待量化模型实例     | 可选，用于获取正样本。<br>数据类型：PyTorch模型                  |


## API接口说明
### add_custormized_dataset_processor
```python
# 添加用户自定义数据集接口，需在set_sample_size(sample_size)之前调用，可选。
# 输入dataset_name: 用户自定义数据集名称，数据类型为string，应与set_sample_size(sample_size)中的用户自定义数据集名称保持一致
# 输入processor：用户自定义数据集处理类实例，继承自DatasetProcessorBase类，
#               需重写DatasetProcessorBase.process_data(indexs)和DatasetProcessorBase.verify_positive_prompt(prompts, labels)方法
CalibrationData.add_custormized_dataset_processor(dataset_name=customized_dataset_name, processor=customized_processor)
```

### set_sample_size
```python
# 设置采样数量，必选。
# 输入sample_size，数据类型dict，如sample_size = {"dataset_name": size}
#               dataset_name数据类型为string，需为config.json文件中配置的数据集名称，或为用户自定义数据集名称，大小写敏感；
#               size数据类型为int，取值范围大于0。若为空或0值，则返回混合校准集对应数据集的采样数量为0；若为非int，则报错；若大于数据集大小或超过可采样数量，则以最大采样数量为准
CalibrationData.set_sample_size(sample_size=sample_size)
```

### set_batch_size
```python
# 设置batch数量，可选。
# 输入batch_size，数据类型为int，取值范围大于0，默认为1
CalibrationData.set_batch_size(batch_size=batch_size)
```

### set_shuffle_seed
```python
# 设置随机种子，在同一设备上可用于复现相同结果，可选。
# 输入shuffle_seed，数据类型为int，默认为0
CalibrationData.set_shuffle_seed(shuffle_seed=shuffle_seed)
```

### process
```python
# 运行接口，必选。
# 输出为 mixed_dataset，数据类型为LIST，如 [{"prompt": prompt1, "ans": ans1}, {"prompt": prompt2, "ans": ans2}]
CalibrationData.process()
```

## 调用说明
请参考 [混合校准数据集使用方法说明](../../../feature_guide/traditional_quantization_v0/foundation_model_quantization_and_calibration.md#混合校准数据集使用方法说明)
