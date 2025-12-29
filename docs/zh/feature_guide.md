# 功能指南
msModelSlim当前支持两种量化服务：V0量化服务与V1量化服务。

msModelSlim V0量化服务基于旧版msModelSlim量化框架及其Python API 接口实现量化功能，将量化过程分为模型加载、离群值抑制和量化校准与保存三个阶段，可以在离群值抑制和量化校准阶段分别采用一种算法。

随着量化算法日益丰富，模型愈发庞大复杂，msModelSlim认识到仅凭离群值抑制算法和量化算法描述的量化方案无法满足新的模型量化需求，在算法之上还有量化策略，即不同的结构可以采用不同的量化算法；同时，随着模型规模愈发庞大，如何利用受限的资源完成大模型的量化也成为迫在眉睫的问题。

msModelSlim认为量化本质上是对模型局部结构的修改和替换，基于此msModelSlim重新设计了量化框架及其Python API，新框架将局部模块、一个算法和一批数据的结合作为基本校准单元，将量化过程视为一系列的基本校准单元，并搭建了msModelSlim V1 量化服务。

当前使用msModelSlim的方式主要包括:
- [大模型一键量化](#大模型一键量化)：通过V0量化服务或V1量化服务实现，用户指定必要参数即可通过命令行快速完成量化；
- [大模型量化敏感层分析](#大模型量化敏感层分析)：通过V1量化服务实现，用户指定必要参数即可通过命令行快速完成量化敏感层分析；
- [大模型脚本量化](#大模型脚本量化)：通过V0量化服务实现，用户按基本量化流程搭建量化脚本，实现模型加载、离群值抑制和量化校准与保存三个阶段完成量化；

用户可快速通过[推荐实践](../../example/README.md)找到以上方式在已支持模型上的实现，快速完成量化；也可以通过下面的表格内容找到各个功能模块的使用说明，自定义完成量化。

### 大模型一键量化


<table>
  <thead>
    <tr>
      <th>框架</th>
      <th>主要模块</th>
      <th>子模块</th>
      <th>功能/主题</th>
      <th>文档</th>
      <th>相关接口说明</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td rowspan="11"><strong>PyTorch</strong></td>
      <td rowspan="10">一键量化</td>
      <td>命令行一键量化</td>
      <td>大模型训练后量化</td>
      <td><a href="feature_guide/quick_quantization/usage.md">一键量化使用说明</a></td>
      <td>
        <a href="feature_guide/quick_quantization/usage.md#接口说明">一键量化接口说明</a>
      </td>
    </tr>
    <tr>
      <td rowspan="9">大模型量化算法</td>
      <td>异常值抑制算法<br>
      Flex Smooth Quant</td>
      <td><a href="algorithms_instruction/flex_smooth_quant.md">Flex Smooth Quant 算法说明</a></td>
      <td>
      -
      </td>
    </tr>
    <tr>
      <td>异常值抑制算法<br>Iterative Smooth</td>
      <td><a href="algorithms_instruction/iterative_smooth.md">Iterative Smooth 算法说明</a></td>
      <td>-</td>
    </tr>
    <tr>
      <td>权重量化算法<br>SSZ</td>
      <td><a href="algorithms_instruction/ssz.md">SSZ 算法说明</a></td>
      <td>-</td>
    </tr>
    <tr>
      <td>异常值抑制算法<br>KV Smooth</td>
      <td><a href="algorithms_instruction/kv_smooth.md">KV Smooth 算法说明</a></td>
      <td>-</td>
    </tr>
    <tr>
      <td>KVCache 量化算法</td>
      <td><a href="algorithms_instruction/kvcache_quant.md">KVCache 量化算法说明</a></td>
      <td>-</td>
    </tr>
    <tr>
      <td>FA3 量化算法</td>
      <td><a href="algorithms_instruction/fa3_quant.md">FA3 量化算法说明</a></td>
      <td>-</td>
    </tr>
    <tr>
      <td>直方图激活量化算法</td>
      <td><a href="algorithms_instruction/histogram_activation_quantization.md">直方图激活量化算法说明</a></td>
      <td>-</td>
    </tr>
    <tr>
      <td>激活值阶段间混合量化算法<br>PDMIX</td>
      <td><a href="algorithms_instruction/pdmix.md">PDMIX 算法说明</a></td>
      <td>-</td>
    </tr>
    <tr>
      <td>大模型浮点稀疏</td>
      <td><a href="algorithms_instruction/float_sparse.md">大模型浮点稀疏</a>
      <td>-</td>
    </tr>
  </tbody>
</table>

### 大模型量化敏感层分析

<table>
  <thead>
    <tr>
      <th>框架</th>
      <th>模块</th>
      <th>功能/主题</th>
      <th>文档</th>
      <th>相关接口说明</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><strong>PyTorch</strong></td>
      <td>量化敏感层分析</td>
      <td>分析量化过程中的精度敏感层</td>
      <td><a href="feature_guide/quantization_sensitive_layer_analysis/analyze_api_usage.md">量化敏感层分析使用指南</a></td>
      <td>
        <a href="feature_guide/quantization_sensitive_layer_analysis/analyze_api_usage.md#必需参数">接口说明：必需参数</a><br>
        <a href="feature_guide/quantization_sensitive_layer_analysis/analyze_api_usage.md#可选参数">接口说明：可选参数</a>
      </td>
    </tr>
  </tbody>
</table>

### 大模型脚本量化

<table>
  <thead>
    <tr>
      <th>框架</th>
      <th>模块</th>
      <th>功能/主题</th>
      <th>文档</th>
      <th>相关接口说明</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td rowspan="9"><strong>PyTorch</strong></td>
      <td rowspan="6">大模型量化</td>
      <td>大模型训练后量化</td>
      <td><a href="feature_guide/scripts_based_quantization_and_other_features/pytorch/foundation_model_post_training_quantization.md">大模型训练后量化</a></td>
      <td>
        <a href="python_api/foundation_model_compression_apis/foundation_model_quantization_apis/AntiOutlierConfig.md">AntiOutlierConfig</a><br>
        <a href="python_api/foundation_model_compression_apis/foundation_model_quantization_apis/AntiOutlier.md">AntiOutlier</a><br>
        <a href="python_api/foundation_model_compression_apis/foundation_model_quantization_apis/process().md">process</a><br>
        <a href="python_api/foundation_model_compression_apis/foundation_model_quantization_apis/pytorch_QuantConfig.md">QuantConfig</a><br>
        <a href="python_api/foundation_model_compression_apis/foundation_model_quantization_apis/pytorch_Calibrator.md">Calibrator</a><br>
        <a href="python_api/foundation_model_compression_apis/foundation_model_quantization_apis/pytorch_run().md">run</a><br>
        <a href="python_api/foundation_model_compression_apis/foundation_model_quantization_apis/pytorch_save().md">save</a><br>
        <a href="python_api/foundation_model_compression_apis/foundation_model_quantization_apis/LayerSelector.md">LayerSelector</a><br>                
        <a href="python_api/foundation_model_compression_apis/foundation_model_quantization_apis/FakeQuantizeCalibrator.md">FakeQuantizeCalibrator</a><br>
      </td>
    </tr>
    <tr>
      <td>FA 量化使用说明</td>
      <td><a href="feature_guide/scripts_based_quantization_and_other_features/pytorch/fa_quantization_usage.md">FA量化使用说明</a></td>
      <td><a href="python_api/foundation_model_compression_apis/foundation_model_quantization_apis/pytorch_FAQuantizer.md">FAQuantizer</a><br><a href="python_api/foundation_model_compression_apis/foundation_model_quantization_apis/pytorch_quant().md">quant</a><br></td>
    </tr>
    <tr>
      <td>低显存量化特性使用说明</td>
      <td><a href="feature_guide/scripts_based_quantization_and_other_features/pytorch/low_vram_quantization_usage.md">低显存量化特性使用说明</a></td>
      <td>-</td>
    </tr>
    <tr>
      <td>混合校准数据集</td>
      <td><a href="feature_guide/scripts_based_quantization_and_other_features/pytorch/mixed_calibration_dataset.md">混合校准数据集</a></td>
      <td>
      <a href="python_api/foundation_model_compression_apis/foundation_model_quantization_apis/CalibrationData.md">CalibrationData</a><br>
      </td>
    </tr>
    <tr>
      <td>MindSpeed 适配器</td>
      <td><a href="feature_guide/scripts_based_quantization_and_other_features/pytorch/mindspeed_adapter.md">MindSpeed适配器</a></td>
      <td>
      <a href="python_api/foundation_model_compression_apis/foundation_model_quantization_apis/mindspeed_ModelAdapter.md">ModelAdapter</a><br>
      <a href="python_api/foundation_model_compression_apis/foundation_model_quantization_apis/mindspeed_AntiOutlierAdapter.md">AntiOutlierAdapter</a><br>
      <a href="python_api/foundation_model_compression_apis/foundation_model_quantization_apis/mindspeed_CalibratorAdapter.md">CalibratorAdapter</a><br>
      <a href="python_api/foundation_model_compression_apis/foundation_model_quantization_apis/mindspeed_process().md">process()</a><br>
      </td>
    </tr>
    <tr>
      <td>开源权重转换为 msModelSlim 权重</td>
      <td><a href="feature_guide/scripts_based_quantization_and_other_features/pytorch/convert_opensource_weights_to_msmodelslim_format.md">开源权重转换为msModelSlim权重</a></td>
      <td>-</td>
    </tr>
    <tr>
      <td rowspan="2">大模型稀疏量化</td>
      <td>大模型稀疏量化</td>
      <td><a href="feature_guide/scripts_based_quantization_and_other_features/pytorch/foundation_model_sparsification_quantization.md">大模型稀疏量化</a></td>
      <td>-</td>
    </tr>
    <tr>
      <td>权重压缩</td>
      <td><a href="feature_guide/scripts_based_quantization_and_other_features/pytorch/weight_compression.md">权重压缩</a></td>
      <td>
        <a href="python_api/foundation_model_compression_apis/weight_compress_apis/CompressConfig.md">CompressConfig</a><br>
        <a href="python_api/foundation_model_compression_apis/weight_compress_apis/Compressor.md">Compressor</a>
        <a href="python_api/foundation_model_compression_apis/weight_compress_apis/run().md">run</a><br>
        <a href="python_api/foundation_model_compression_apis/weight_compress_apis/export().md">export</a><br>
        <a href="python_api/foundation_model_compression_apis/weight_compress_apis/export_safetensors().md">export_safetensors</a><br>
      </td>
    </tr>
    <tr>
      <td rowspan="2">多模态生成模型推理优化</td>
      <td>多模态生成模型推理优化</td>
      <td><a href="feature_guide/scripts_based_quantization_and_other_features/pytorch/inference_optimization_for_multimodal_generative_model.md">多模态生成模型推理优化</a></td>
      <td>
      <a href="python_api/multimodal_inference_apis/DitCache/DitCacheSearchConfig.md">DitCache: DitCacheSearchConfig</a><br>
      <a href="python_api/multimodal_inference_apis/DitCache/DitCacheAdaptor.md">DitCache: DitCacheAdaptor</a><br>
      <a href="python_api/multimodal_inference_apis/sampling_optimization_apis/ReStepSearchConfig.md">采样优化接口: ReStepSearchConfig</a><br>
      <a href="python_api/multimodal_inference_apis/sampling_optimization_apis/ReStepAdaptor.md">采样优化接口: ReStepAdaptor</a><br>
      </td>
    </tr>
  </tbody>
</table>

<details>
<summary><strong>查看其他功能</strong></summary>

### 其他功能

#### PyTorch

<table>
  <thead>
    <tr>
      <th>框架</th>
      <th>模块</th>
      <th>功能/主题</th>
      <th>文档</th>
      <th>相关接口说明</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td rowspan="11"><strong>PyTorch</strong></td>
      <td>大模型压缩</td>
      <td>长序列压缩</td>
      <td><a href="feature_guide/scripts_based_quantization_and_other_features/pytorch/long_sequence_compression.md">长序列压缩</a></td>
      <td>
      <a href="python_api/foundation_model_compression_apis/long_sequence_compression_apis/RACompressConfig.md">Alibi编码类型: RACompressConfig</a><br>
      <a href="python_api/foundation_model_compression_apis/long_sequence_compression_apis/RACompressor.md">Alibi编码类型: RACompressor</a><br>
      <a href="python_api/foundation_model_compression_apis/long_sequence_compression_apis/get_alibi_windows.md">Alibi编码类型: get_alibi_windows</a><br>
      <a href="python_api/foundation_model_compression_apis/long_sequence_compression_apis/RARopeCompressConfig.md">RoPE编码类型: RARopeCompressConfig</a><br>
      <a href="python_api/foundation_model_compression_apis/long_sequence_compression_apis/RARopeCompressor.md">RoPE编码类型: RARopeCompressor</a><br>
      <a href="python_api/foundation_model_compression_apis/long_sequence_compression_apis/get_compress_heads.md">RoPE编码类型: get_compress_heads</a><br>      
      </td>
    </tr>
    <tr>
      <td>伪量化精度测试</td>
      <td>伪量化精度测试工具</td>
      <td><a href="feature_guide/scripts_based_quantization_and_other_features/pytorch/fake_quantization_accuracy_testing_tool.md">伪量化精度测试工具</a></td>
      <td>-</td>
    </tr>
    <tr>
      <td rowspan="2">常规模型量化</td>
      <td>训练后量化</td>
      <td><a href="feature_guide/scripts_based_quantization_and_other_features/pytorch/post_training_quantization.md">训练后量化</a></td>
      <td>
        <a href="python_api/quantization_apis/[pytorch]post_training_quantization/QuantConfig.md">QuantConfig</a><br>
        <a href="python_api/quantization_apis/[pytorch]post_training_quantization/Calibrator.md">Calibrator</a><br>
        <a href="python_api/quantization_apis/[pytorch]post_training_quantization/get_quant_params.md">get_quant_params</a><br>
        <a href="python_api/quantization_apis/[pytorch]post_training_quantization/export_param.md">export_param</a><br>
        <a href="python_api/quantization_apis/[pytorch]post_training_quantization/export_quant_safetensor.md">export_quant_safetensor</a><br>
        <a href="python_api/quantization_apis/[pytorch]post_training_quantization/export_quant_onnx.md">export_quant_onnx</a><br>
      </td>
    </tr>
    <tr>
      <td>量化感知训练</td>
      <td><a href="feature_guide/scripts_based_quantization_and_other_features/pytorch/quantization_aware_training.md">量化感知训练</a></td>
      <td>
        <a href="python_api/quantization_apis/quantization_aware_training/QatConfig.md">QatConfig</a><br>
        <a href="python_api/quantization_apis/quantization_aware_training/qsin_qat.md">qsin_qat</a><br>
        <a href="python_api/quantization_apis/quantization_aware_training/save_qsin_qat_model.md">save_qsin_qat_model</a><br>
      </td>
    </tr>
    <tr>
      <td rowspan="2">模型稀疏</td>
      <td>模型稀疏</td>
      <td><a href="feature_guide/scripts_based_quantization_and_other_features/pytorch/model_sparsification.md">模型稀疏</a></td>
      <td>
      <a href="python_api/foundation_model_compression_apis/foundation_model_sparsification_apis/SparseConfig.md">SparseConfig</a><br>
      <a href="python_api/foundation_model_compression_apis/foundation_model_sparsification_apis/Compressor.md">Compressor</a><br>
      <a href="python_api/foundation_model_compression_apis/foundation_model_sparsification_apis/compress().md">compress()</a><br>
      </td>
    </tr>
    <tr>
      <td>稀疏加速训练</td>
      <td><a href="feature_guide/scripts_based_quantization_and_other_features/pytorch/sparse_acceleration_training.md">稀疏加速训练</a></td>
      <td>
        <a href="python_api/sparse_acceleration_training_apis/sparse_model_depth.md">sparse_model_depth</a><br>
        <a href="python_api/sparse_acceleration_training_apis/sparse_model_width.md">sparse_model_width</a>
      </td>
    </tr>
    <tr>
      <td rowspan="2">模型剪枝</td>
      <td>Transformer 类模型权重剪枝调优</td>
      <td><a href="feature_guide/scripts_based_quantization_and_other_features/pytorch/transformer_model_pruning_and_tuning.md">Transformer类模型权重剪枝调优</a></td>
      <td>
        <a href="python_api/pruning_apis/PruneConfig/add_blocks_params.md">add_blocks_params</a><br>
        <a href="python_api/pruning_apis/PruneConfig/set_steps.md">set_steps</a><br>
        <a href="python_api/pruning_apis/prune_model_weight.md">prune_model_weight</a><br>
      </td>
    </tr>
    <tr>
      <td>基于重要性评估的剪枝调优</td>
      <td><a href="feature_guide/scripts_based_quantization_and_other_features/pytorch/importance_based_pruning_and_tuning.md">基于重要性评估的剪枝调优</a></td>
      <td>
      <a href="python_api/pruning_apis/PruneTorch/__init__.md">__init__</a><br>
      <a href="python_api/pruning_apis/PruneTorch/set_importance_evaluation_function.md">set_importance_evaluation_function</a><br>
      <a href="python_api/pruning_apis/PruneTorch/set_node_reserved_ratio.md">set_node_reserved_ratio</a><br>
      <a href="python_api/pruning_apis/PruneTorch/analysis.md">analysis</a><br>
      <a href="python_api/pruning_apis/PruneTorch/prune.md">prune</a><br>
      <a href="python_api/pruning_apis/PruneTorch/prune_by_desc.md">prune_by_desc</a><br>
      </td>
    </tr>
    <tr>
      <td>模型低秩分解</td>
      <td>模型低秩分解</td>
      <td><a href="feature_guide/scripts_based_quantization_and_other_features/pytorch/model_low_rank_factorization.md">模型低秩分解</a></td>
      <td>
        <a href="python_api/low_rank_decompression_apis/Decompose/__init__.md">__init__</a><br>
        <a href="python_api/low_rank_decompression_apis/Decompose/from_ratio.md">from_ratio</a><br>
        <a href="python_api/low_rank_decompression_apis/Decompose/from_vbmf.md">from_vbmf</a><br>
        <a href="python_api/low_rank_decompression_apis/Decompose/from_dict.md">from_dict</a><br>
        <a href="python_api/low_rank_decompression_apis/Decompose/from_file.md">from_file</a><br>
        <a href="python_api/low_rank_decompression_apis/Decompose/from_fixed.md">from_fixed</a><br>
        <a href="python_api/low_rank_decompression_apis/Decompose/decompose_network.md">decompose_network</a><br>
        <a href="python_api/low_rank_decompression_apis/count_parameters.md">count_parameters</a><br>
      </td>
    </tr>
  </tbody>
</table>

#### MindSpore

<table>
  <thead>
    <tr>
      <th>框架</th>
      <th>模块</th>
      <th>文档</th>
      <th>相关接口说明</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><strong>MindSpore</strong></td>
      <td>常规模型训练后量化</td>
      <td><a href="feature_guide/scripts_based_quantization_and_other_features/mindspore/post_training_quantization.md">训练后量化</a></td>
      <td>
      <a href="python_api/quantization_apis/[mindspore]post_training_quantization/create_quant_config.md">create_quant_config</a><br>
      <a href="python_api/quantization_apis/[mindspore]post_training_quantization/quantize_model.md">quantize_model</a><br>
      <a href="python_api/quantization_apis/[mindspore]post_training_quantization/save_model.md">save_model</a><br>
      </td>
    </tr>
  </tbody>
</table>

#### ONNX

<table>
  <thead>
    <tr>
      <th>框架</th>
      <th>模块</th>
      <th>文档</th>
      <th>相关接口说明</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><strong>ONNX</strong></td>
      <td>常规模型训练后量化</td>
      <td><a href="feature_guide/scripts_based_quantization_and_other_features/onnx/post_training_quantization.md">训练后量化</a></td>
      <td>
      <a href="python_api/quantization_apis/[onnx]post_training_quantization/QuantConfig_post_training_quant.md">post_training_quant接口: QuantConfig</a><br>
      <a href="python_api/quantization_apis/[onnx]post_training_quantization/preprocess_func_coco.md">post_training_quant接口: preprocess_func_coco</a><br>
      <a href="python_api/quantization_apis/[onnx]post_training_quantization/preprocess_func_imagenet.md">post_training_quant接口: preprocess_func_imagenet</a><br>
      <a href="python_api/quantization_apis/[onnx]post_training_quantization/run_quantize.md">post_training_quant接口: run_quantize</a><br>
      <a href="python_api/quantization_apis/[onnx]post_training_quantization/QuantConfig_squant_ptq.md">squant_ptq接口: QuantConfig</a><br>
      <a href="python_api/quantization_apis/[onnx]post_training_quantization/OnnxCalibrator.md">squant_ptq接口: OnnxCalibrator</a><br>
      <a href="python_api/quantization_apis/[onnx]post_training_quantization/run().md">squant_ptq接口: run()</a><br>
      <a href="python_api/quantization_apis/[onnx]post_training_quantization/export_quant_onnx.md">squant_ptq接口: export_quant_onnx</a><br>
      </td>
    </tr>
  </tbody>
</table>

#### common

<table>
  <thead>
    <tr>
      <th>类别</th>
      <th>模块</th>
      <th>文档</th>
      <th>相关接口说明</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><strong>common</strong></td>
      <td>模型蒸馏</td>
      <td><a href="feature_guide/scripts_based_quantization_and_other_features/common/model_distillation.md">模型蒸馏</a></td>
      <td>
        <a href="python_api/distillation_apis/get_distill_model.md">get_distill_model</a><br>
        <a href="python_api/distillation_apis/KnowledgeDistillConfig/set_teacher_train.md">set_teacher_train</a><br>
        <a href="python_api/distillation_apis/KnowledgeDistillConfig/add_inter_soft_label.md">add_inter_soft_label</a><br>
        <a href="python_api/distillation_apis/KnowledgeDistillConfig/add_output_soft_label.md">add_output_soft_label</a><br>
        <a href="python_api/distillation_apis/KnowledgeDistillConfig/set_hard_label.md">set_hard_label</a><br>
        <a href="python_api/distillation_apis/KnowledgeDistillConfig/add_custom_loss_func.md">add_custom_loss_func</a>
      </td>
    </tr>
  </tbody>
</table>

</details>