# msModelSlim 资料

欢迎来到 msModelSlim 资料，此资料目录将为你提供有关 msModelSlim 快速入门、功能、算法、接口、FAQ等介绍，帮助你更快熟悉 msModelSlim 工具。

msModelSlim 资料将在工具迭代更新中不断完善，有任何问题请联系我们，让我们一起构筑更易用的 msModelSlim 工具！

## 安装指南

具体安装步骤请查看[安装指南](./安装指南.md)

## 快速入门


| 名称             | 文档                                                 |
| ------------------ | ------------------------------------------------------ |
| 一键量化快速入门 | [一键量化快速入门](./快速入门/一键量化快速入门.md) |

## 支持矩阵


| 类别           | 文档                                  |
| ---------------- | --------------------------------------- |
| 大模型支持矩阵 | [大模型支持矩阵](./支持矩阵/大模型支持矩阵.md) |

## 功能指南
msModelSlim当前支持两种量化服务：V0量化服务与V1量化服务。

msModelSlim V0量化服务基于旧版msModelSlim量化框架及其Python API 接口实现量化功能，将量化过程分为模型加载、离群值抑制和量化校准与保存三个阶段，可以在离群值抑制和量化校准阶段分别采用一种算法。

随着量化算法日益丰富，模型愈发庞大复杂，msModelSlim认识到仅凭离群值抑制算法和量化算法描述的量化方案无法满足新的模型量化需求，在算法之上还有量化策略，即不同的结构可以采用不同的量化算法；同时，随着模型规模愈发庞大，如何利用受限的资源完成大模型的量化也成为迫在眉睫的问题。

msModelSlim认为量化本质上是对模型局部结构的修改和替换，基于此msModelSlim重新设计了量化框架及其Python API，新框架将局部模块、一个算法和一批数据的结合作为基本校准单元，将量化过程视为一系列的基本校准单元，并搭建了msModelSlim V1 量化服务。

当前使用msModelSlim的方式主要包括:
- [大模型一键量化](#大模型一键量化)：通过V0量化服务或V1量化服务实现，用户指定必要参数即可通过命令行快速完成量化；
- [大模型量化敏感层分析](#大模型量化敏感层分析)：通过V1量化服务实现，用户指定必要参数即可通过命令行快速完成量化敏感层分析；
- [大模型脚本量化](#大模型脚本量化)：通过V0量化服务实现，用户按基本量化流程搭建量化脚本，实现模型加载、离群值抑制和量化校准与保存三个阶段完成量化；

用户可快速通过[推荐实践](../example/README.md)找到以上方式在已支持模型上的实现，快速完成量化；也可以通过下面的表格内容找到各个功能模块的使用说明，自定义完成量化。

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
      <td><a href="./功能指南/一键量化/使用说明.md">一键量化使用说明</a></td>
      <td>
        <a href="./功能指南/一键量化/使用说明.md#接口说明">一键量化接口说明</a>
      </td>
    </tr>
    <tr>
      <td rowspan="9">大模型量化算法</td>
      <td>异常值抑制算法<br>
      Flex Smooth Quant</td>
      <td><a href="./算法说明/Flex_Smooth_Quant.md">Flex Smooth Quant 算法说明</a></td>
      <td>
      -
      </td>
    </tr>
    <tr>
      <td>异常值抑制算法<br>Iterative Smooth</td>
      <td><a href="./算法说明/Iterative_Smooth.md">Iterative Smooth 算法说明</a></td>
      <td>-</td>
    </tr>
    <tr>
      <td>权重量化算法<br>SSZ</td>
      <td><a href="./算法说明/ssz.md">SSZ 算法说明</a></td>
      <td>-</td>
    </tr>
    <tr>
      <td>异常值抑制算法<br>KV Smooth</td>
      <td><a href="./算法说明/kv_smooth.md">KV Smooth 算法说明</a></td>
      <td>-</td>
    </tr>
    <tr>
      <td>KVCache 量化算法</td>
      <td><a href="./算法说明/KVCache_quant.md">KVCache 量化算法说明</a></td>
      <td>-</td>
    </tr>
    <tr>
      <td>FA3 量化算法</td>
      <td><a href="./算法说明/FA3_quant.md">FA3 量化算法说明</a></td>
      <td>-</td>
    </tr>
    <tr>
      <td>直方图激活量化算法</td>
      <td><a href="./算法说明/histogram_activation_quantization.md">直方图激活量化算法说明</a></td>
      <td>-</td>
    </tr>
    <tr>
      <td>激活值阶段间混合量化算法<br>PDMIX</td>
      <td><a href="./算法说明/pdmix.md">PDMIX 算法说明</a></td>
      <td>-</td>
    </tr>
    <tr>
      <td>大模型浮点稀疏</td>
      <td><a href="./算法说明/float_sparse.md">大模型浮点稀疏</a>
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
      <td><a href="./功能指南/量化敏感层分析/analyze接口使用指南.md">量化敏感层分析使用指南</a></td>
      <td>
        <a href="./功能指南/量化敏感层分析/analyze接口使用指南.md#必需参数">接口说明：必需参数</a><br>
        <a href="./功能指南/量化敏感层分析/analyze接口使用指南.md#可选参数">接口说明：可选参数</a>
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
      <td><a href="./功能指南/脚本量化与其他功能/pytorch/llm_ptq/大模型训练后量化.md">大模型训练后量化</a></td>
      <td>
        <a href="./接口说明/Python-API接口说明/大模型压缩接口/大模型量化接口/AntiOutlierConfig.md">AntiOutlierConfig</a><br>
        <a href="./接口说明/Python-API接口说明/大模型压缩接口/大模型量化接口/AntiOutlier.md">AntiOutlier</a><br>
        <a href="./接口说明/Python-API接口说明/大模型压缩接口/大模型量化接口/process().md">process</a><br>
        <a href="./接口说明/Python-API接口说明/大模型压缩接口/大模型量化接口/PyTorch/QuantConfig.md">QuantConfig</a><br>
        <a href="./接口说明/Python-API接口说明/大模型压缩接口/大模型量化接口/PyTorch/Calibrator.md">Calibrator</a><br>
        <a href="./接口说明/Python-API接口说明/大模型压缩接口/大模型量化接口/PyTorch/run().md">run</a><br>
        <a href="./接口说明/Python-API接口说明/大模型压缩接口/大模型量化接口/PyTorch/save().md">save</a><br>
        <a href="./接口说明/Python-API接口说明/大模型压缩接口/大模型量化接口/LayerSelector.md">LayerSelector</a><br>                
        <a href="./接口说明/Python-API接口说明/大模型压缩接口/大模型量化接口/FakeQuantizeCalibrator.md">FakeQuantizeCalibrator</a><br>
      </td>
    </tr>
    <tr>
      <td>FA 量化使用说明</td>
      <td><a href="./功能指南/脚本量化与其他功能/pytorch/llm_ptq/FA量化使用说明.md">FA量化使用说明</a></td>
      <td><a href="./接口说明/Python-API接口说明/大模型压缩接口/大模型量化接口/PyTorch/FAQuantizer.md">FAQuantizer</a><br><a href="./接口说明/Python-API接口说明/大模型压缩接口/大模型量化接口/PyTorch/quant().md">quant</a><br></td>
    </tr>
    <tr>
      <td>低显存量化特性使用说明</td>
      <td><a href="./功能指南/脚本量化与其他功能/pytorch/llm_ptq/低显存量化特性使用说明.md">低显存量化特性使用说明</a></td>
      <td>-</td>
    </tr>
    <tr>
      <td>混合校准数据集</td>
      <td><a href="./功能指南/脚本量化与其他功能/pytorch/llm_ptq/混合校准数据集.md">混合校准数据集</a></td>
      <td>
      <a href="./接口说明/Python-API接口说明/大模型压缩接口/大模型量化接口/CalibrationData.md">CalibrationData</a><br>
      </td>
    </tr>
    <tr>
      <td>MindSpeed 适配器</td>
      <td><a href="./功能指南/脚本量化与其他功能/pytorch/llm_ptq/MindSpeed适配器.md">MindSpeed适配器</a></td>
      <td>
      <a href=".//接口说明/Python-API接口说明/大模型压缩接口/大模型量化接口/PyTorch/mindspeed/ModelAdapter.md">ModelAdapter</a><br>
      <a href=".//接口说明/Python-API接口说明/大模型压缩接口/大模型量化接口/PyTorch/mindspeed/AntiOutlierAdapter.md">AntiOutlierAdapter</a><br>
      <a href=".//接口说明/Python-API接口说明/大模型压缩接口/大模型量化接口/PyTorch/mindspeed/CalibratorAdapter.md">CalibratorAdapter</a><br>
      <a href=".//接口说明/Python-API接口说明/大模型压缩接口/大模型量化接口/PyTorch/mindspeed/process().md">process()</a><br>
      </td>
    </tr>
    <tr>
      <td>开源权重转换为 msModelSlim 权重</td>
      <td><a href="./功能指南/脚本量化与其他功能/pytorch/llm_ptq/开源权重转换为msModelSlim权重.md">开源权重转换为msModelSlim权重</a></td>
      <td>-</td>
    </tr>
    <tr>
      <td rowspan="2">大模型稀疏量化</td>
      <td>大模型稀疏量化</td>
      <td><a href="./功能指南/脚本量化与其他功能/pytorch/llm_sparsequant/大模型稀疏量化.md">大模型稀疏量化</a></td>
      <td>-</td>
    </tr>
    <tr>
      <td>权重压缩</td>
      <td><a href="./功能指南/脚本量化与其他功能/pytorch/weight_compression/权重压缩.md">权重压缩</a></td>
      <td>
        <a href="./接口说明/Python-API接口说明/大模型压缩接口/权重压缩接口/CompressConfig.md">CompressConfig</a><br>
        <a href="./接口说明/Python-API接口说明/大模型压缩接口/权重压缩接口/Compressor.md">Compressor</a>
        <a href="./接口说明/Python-API接口说明/大模型压缩接口/权重压缩接口/run().md">run</a><br>
        <a href="./接口说明/Python-API接口说明/大模型压缩接口/权重压缩接口/export().md">export</a><br>
        <a href="./接口说明/Python-API接口说明/大模型压缩接口/权重压缩接口/export_safetensors().md">export_safetensors</a><br>
      </td>
    </tr>
    <tr>
      <td rowspan="2">多模态生成模型推理优化</td>
      <td>多模态生成模型推理优化</td>
      <td><a href="./功能指南/脚本量化与其他功能/pytorch/multimodal_sd/多模态生成模型推理优化.md">多模态生成模型推理优化</a></td>
      <td>
      <a href="./接口说明/Python-API接口说明/多模态推理优化接口/DitCache/DitCacheSearchConfig.md">DitCache: DitCacheSearchConfig</a><br>
      <a href="./接口说明/Python-API接口说明/多模态推理优化接口/DitCache/DitCacheAdaptor.md">DitCache: DitCacheAdaptor</a><br>
      <a href="./接口说明/Python-API接口说明/多模态推理优化接口/采样优化接口/ReStepSearchConfig.md">采样优化接口: ReStepSearchConfig</a><br>
      <a href="./接口说明/Python-API接口说明/多模态推理优化接口/采样优化接口/ReStepAdaptor.md">采样优化接口: ReStepAdaptor</a><br>
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
      <td><a href="./功能指南/脚本量化与其他功能/pytorch/ra_compression/长序列压缩.md">长序列压缩</a></td>
      <td>
      <a href="./接口说明/Python-API接口说明/大模型压缩接口/长序列压缩接口/Alibi编码类型/RACompressConfig.md">Alibi编码类型: RACompressConfig</a><br>
      <a href="./接口说明/Python-API接口说明/大模型压缩接口/长序列压缩接口/Alibi编码类型/RACompressor.md">Alibi编码类型: RACompressor</a><br>
      <a href="./接口说明/Python-API接口说明/大模型压缩接口/长序列压缩接口/Alibi编码类型/get_alibi_windows.md">Alibi编码类型: get_alibi_windows</a><br>
      <a href="./接口说明/Python-API接口说明/大模型压缩接口/长序列压缩接口/RoPE编码类型/RARopeCompressConfig.md">RoPE编码类型: RARopeCompressConfig</a><br>
      <a href="./接口说明/Python-API接口说明/大模型压缩接口/长序列压缩接口/RoPE编码类型/RARopeCompressor.md">RoPE编码类型: RARopeCompressor</a><br>
      <a href="./接口说明/Python-API接口说明/大模型压缩接口/长序列压缩接口/RoPE编码类型/get_compress_heads.md">RoPE编码类型: get_compress_heads</a><br>      
      </td>
    </tr>
    <tr>
      <td>伪量化精度测试</td>
      <td>伪量化精度测试工具</td>
      <td><a href="./功能指南/脚本量化与其他功能/pytorch/precision_tool/伪量化精度测试工具.md">伪量化精度测试工具</a></td>
      <td>-</td>
    </tr>
    <tr>
      <td rowspan="2">常规模型量化</td>
      <td>训练后量化</td>
      <td><a href="./功能指南/脚本量化与其他功能/pytorch/quant/训练后量化.md">训练后量化</a></td>
      <td>
        <a href="./接口说明/Python-API接口说明/量化接口/训练后量化（PyTorch）/QuantConfig.md">QuantConfig</a><br>
        <a href="./接口说明/Python-API接口说明/量化接口/训练后量化（PyTorch）/Calibrator.md">Calibrator</a><br>
        <a href="./接口说明/Python-API接口说明/量化接口/训练后量化（PyTorch）/get_quant_params.md">get_quant_params</a><br>
        <a href="./接口说明/Python-API接口说明/量化接口/训练后量化（PyTorch）/export_param.md">export_param</a><br>
        <a href="./接口说明/Python-API接口说明/量化接口/训练后量化（PyTorch）/export_quant_safetensor.md">export_quant_safetensor</a><br>
        <a href="./接口说明/Python-API接口说明/量化接口/训练后量化（PyTorch）/export_quant_onnx.md">export_quant_onnx</a><br>
      </td>
    </tr>
    <tr>
      <td>量化感知训练</td>
      <td><a href="./功能指南/脚本量化与其他功能/pytorch/quant/量化感知训练.md">量化感知训练</a></td>
      <td>
        <a href="./接口说明/Python-API接口说明/量化接口/量化感知训练/QatConfig.md">QatConfig</a><br>
        <a href="./接口说明/Python-API接口说明/量化接口/量化感知训练/qsin_qat.md">qsin_qat</a><br>
        <a href="./接口说明/Python-API接口说明/量化接口/量化感知训练/save_qsin_qat_model.md">save_qsin_qat_model</a><br>
      </td>
    </tr>
    <tr>
      <td rowspan="2">模型稀疏</td>
      <td>模型稀疏</td>
      <td><a href="./功能指南/脚本量化与其他功能/pytorch/sparse/模型稀疏.md">模型稀疏</a></td>
      <td>
      <a href="./接口说明/Python-API接口说明/大模型压缩接口/大模型稀疏接口/SparseConfig.md">SparseConfig</a><br>
      <a href="./接口说明/Python-API接口说明/大模型压缩接口/大模型稀疏接口/Compressor.md">Compressor</a><br>
      <a href="./接口说明/Python-API接口说明/大模型压缩接口/大模型稀疏接口/compress().md">compress()</a><br>
      </td>
    </tr>
    <tr>
      <td>稀疏加速训练</td>
      <td><a href="./功能指南/脚本量化与其他功能/pytorch/sparse/稀疏加速训练.md">稀疏加速训练</a></td>
      <td>
        <a href="./接口说明/Python-API接口说明/稀疏加速训练接口/sparse_model_depth.md">sparse_model_depth</a><br>
        <a href="./接口说明/Python-API接口说明/稀疏加速训练接口/sparse_model_width.md">sparse_model_width</a>
      </td>
    </tr>
    <tr>
      <td rowspan="2">模型剪枝</td>
      <td>Transformer 类模型权重剪枝调优</td>
      <td><a href="./功能指南/脚本量化与其他功能/pytorch/prune/Transformer类模型权重剪枝调优.md">Transformer类模型权重剪枝调优</a></td>
      <td>
        <a href="./接口说明/Python-API接口说明/剪枝接口/PruneConfig/add_blocks_params.md">add_blocks_params</a><br>
        <a href="./接口说明/Python-API接口说明/剪枝接口/PruneConfig/set_steps.md">set_steps</a><br>
        <a href="./接口说明/Python-API接口说明/剪枝接口/prune_model_weight.md">prune_model_weight</a><br>
      </td>
    </tr>
    <tr>
      <td>基于重要性评估的剪枝调优</td>
      <td><a href="./功能指南/脚本量化与其他功能/pytorch/prune/基于重要性评估的剪枝调优.md">基于重要性评估的剪枝调优</a></td>
      <td>
      <a href="./接口说明/Python-API接口说明/剪枝接口/PruneTorch/__init__.md">__init__</a><br>
      <a href="./接口说明/Python-API接口说明/剪枝接口/PruneTorch/set_importance_evaluation_function.md">set_importance_evaluation_function</a><br>
      <a href="./接口说明/Python-API接口说明/剪枝接口/PruneTorch/set_node_reserved_ratio.md">set_node_reserved_ratio</a><br>
      <a href="./接口说明/Python-API接口说明/剪枝接口/PruneTorch/analysis.md">analysis</a><br>
      <a href="./接口说明/Python-API接口说明/剪枝接口/PruneTorch/prune.md">prune</a><br>
      <a href="./接口说明/Python-API接口说明/剪枝接口/PruneTorch/prune_by_desc.md">prune_by_desc</a><br>
      </td>
    </tr>
    <tr>
      <td>模型低秩分解</td>
      <td>模型低秩分解</td>
      <td><a href="./功能指南/脚本量化与其他功能/pytorch/low_rank_decompose/模型低秩分解.md">模型低秩分解</a></td>
      <td>
        <a href="./接口说明/Python-API接口说明/低秩分解接口/Decompose/__init__.md">__init__</a><br>
        <a href="./接口说明/Python-API接口说明/低秩分解接口/Decompose/from_ratio.md">from_ratio</a><br>
        <a href="./接口说明/Python-API接口说明/低秩分解接口/Decompose/from_vbmf.md">from_vbmf</a><br>
        <a href="./接口说明/Python-API接口说明/低秩分解接口/Decompose/from_dict.md">from_dict</a><br>
        <a href="./接口说明/Python-API接口说明/低秩分解接口/Decompose/from_file.md">from_file</a><br>
        <a href="./接口说明/Python-API接口说明/低秩分解接口/Decompose/from_fixed.md">from_fixed</a><br>
        <a href="./接口说明/Python-API接口说明/低秩分解接口/Decompose/decompose_network.md">decompose_network</a><br>
        <a href="./接口说明/Python-API接口说明/低秩分解接口/count_parameters.md">count_parameters</a><br>
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
      <td><a href="./功能指南/脚本量化与其他功能/mindspore/quant/训练后量化.md">训练后量化</a></td>
      <td>
      <a href="./接口说明/Python-API接口说明/量化接口/训练后量化（MindSpore）/create_quant_config.md">create_quant_config</a><br>
      <a href="./接口说明/Python-API接口说明/量化接口/训练后量化（MindSpore）/quantize_model.md">quantize_model</a><br>
      <a href="./接口说明/Python-API接口说明/量化接口/训练后量化（MindSpore）/save_model.md">save_model</a><br>
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
      <td><a href="./功能指南/脚本量化与其他功能/onnx/训练后量化.md">训练后量化</a></td>
      <td>
      <a href="./接口说明/Python-API接口说明/量化接口/训练后量化（ONNX）/post_training_quant接口/QuantConfig.md">post_training_quant接口: QuantConfig</a><br>
      <a href="./接口说明/Python-API接口说明/量化接口/训练后量化（ONNX）/post_training_quant接口/preprocess_func_coco.md">post_training_quant接口: preprocess_func_coco</a><br>
      <a href="./接口说明/Python-API接口说明/量化接口/训练后量化（ONNX）/post_training_quant接口/preprocess_func_imagenet.md">post_training_quant接口: preprocess_func_imagenet</a><br>
      <a href="./接口说明/Python-API接口说明/量化接口/训练后量化（ONNX）/post_training_quant接口/run_quantize.md">post_training_quant接口: run_quantize</a><br>
      <a href="./接口说明/Python-API接口说明/量化接口/训练后量化（ONNX）/squant_ptq接口/QuantConfig.md">squant_ptq接口: QuantConfig</a><br>
      <a href="./接口说明/Python-API接口说明/量化接口/训练后量化（ONNX）/squant_ptq接口/OnnxCalibrator.md">squant_ptq接口: OnnxCalibrator</a><br>
      <a href="./接口说明/Python-API接口说明/量化接口/训练后量化（ONNX）/squant_ptq接口/run().md">squant_ptq接口: run()</a><br>
      <a href="./接口说明/Python-API接口说明/量化接口/训练后量化（ONNX）/squant_ptq接口/export_quant_onnx.md">squant_ptq接口: export_quant_onnx</a><br>
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
      <td><a href="./功能指南/脚本量化与其他功能/common/模型蒸馏.md">模型蒸馏</a></td>
      <td>
        <a href="./接口说明/Python-API接口说明/蒸馏接口/get_distill_model.md">get_distill_model</a><br>
        <a href="./接口说明/Python-API接口说明/蒸馏接口/KnowledgeDistillConfig/set_teacher_train.md">set_teacher_train</a><br>
        <a href="./接口说明/Python-API接口说明/蒸馏接口/KnowledgeDistillConfig/add_inter_soft_label.md">add_inter_soft_label</a><br>
        <a href="./接口说明/Python-API接口说明/蒸馏接口/KnowledgeDistillConfig/add_output_soft_label.md">add_output_soft_label</a><br>
        <a href="./接口说明/Python-API接口说明/蒸馏接口/KnowledgeDistillConfig/set_hard_label.md">set_hard_label</a><br>
        <a href="./接口说明/Python-API接口说明/蒸馏接口/KnowledgeDistillConfig/add_custom_loss_func.md">add_custom_loss_func</a>
      </td>
    </tr>
  </tbody>
</table>

</details>

## 自主量化
面向需要将自有模型接入 msModelSlim 的开发者，msModelSlim提供了[自主量化模型接入指南](./自主量化/模型接入.md)。

## 案例集

<table>
  <thead>
    <tr>
      <th>案例分类</th>
      <th>案例名称</th>
      <th>说明</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td rowspan="2"><strong>量化精度调优</strong></td>
      <td>w8a8精度调优策略</td>
      <td><a href="./案例集/w8a8精度调优策略.md">w8a8精度调优策略指南</a></td>
    </tr>
    <tr>
      <td>w8a16精度调优策略</td>
      <td><a href="./案例集/w8a16精度调优策略.md">w8a16精度调优策略指南</a></td>
    </tr>
    <tr>
      <td><strong>稀疏量化调试</strong></td>
      <td>稀疏量化精度调试案例</td>
      <td><a href="./案例集/稀疏量化精度调试案例.md">稀疏量化精度调试方法和案例</a></td>
    </tr>
    <tr>
      <td><strong>代码集成</strong></td>
      <td>量化及稀疏量化场景导入代码样例</td>
      <td><a href="./案例集/量化及稀疏量化场景导入代码样例.md">量化和稀疏量化代码集成示例</a></td>
    </tr>
    <tr>
      <td><strong>权重转换</strong></td>
      <td>msModelSlim量化权重转AutoAWQ&AutoGPTQ使用指南</td>
      <td><a href="./案例集/msModelSlim量化权重转AutoAWQ&AutoGPTQ使用指南.md">量化权重格式转换指南</a></td>
    </tr>
    <tr>
      <td><strong>推理部署</strong></td>
      <td>加速库&MindIE-Torch场景下的量化权重使用案例</td>
      <td><a href="./案例集/加速库&MindIE-Torch场景下的量化权重使用案例.md">推理加速库中量化权重使用方法</a></td>
    </tr>
  </tbody>
</table>


## FAQ

FAQ 旨在帮助用户解决一些使用msModelSlim工具时遇到的常见问题，目前正在逐步完善中，msModelSlim将持续补充和更新。

具体FAQ可查看[FAQ](./FAQ.md)