#  -*- coding: utf-8 -*-
#  Copyright (c) 2025-2025 Huawei Technologies Co., Ltd.
#  #
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#  #
#  http://www.apache.org/licenses/LICENSE-2.0
#  #
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

"""
Interface hub collects all the interfaces in the project to assist the model adapter's development.

Each interface represents a series of demands to model for specific component dedicated to specific domain.
It is unnecessary to implement all the interfaces in a model adapter.
Just implement the interfaces for components you need.

"""

__all__ = [
    # base interface
    'IModel',  # Necessary, including model_type, model_path, trust_remote_code, etc.

    # app interface
    'ModelInfoInterface',  # For Naive Quantization, get model info from model.
    'AnalyzePipelineInterface',  # For Analysis, describing the pipeline of model inference.

    # algorithm interface
    'KVSmoothFusedInterface',  # For KV Smooth, describing the architecture of model.
    'SmoothQuantInterface',  # For Smooth Quant, describing the architecture of model.
    'IterSmoothInterface',  # For Iter Smooth, describing the architecture of model.
    'FlexSmoothQuantInterface',  # For Flex Smooth Quant, describing the architecture of model.

    # remaining interface
    'ModelSlimPipelineInterfaceV0',  # For ModelSlim V0 quant service, describing the pipeline of model inference.
    'MultimodalSDPipelineInterface',  # For MultimodalSD quant service, describing the pipeline of model inference.
    'ModelSlimPipelineInterfaceV1',  # For ModelSlim V1 quant service, describing the pipeline of model inference.

    # FA3 activation quantization interface
    'FA3QuantAdapterInterface', # For FA3 activation quantization, inject placeholders.
    'FA3QuantPlaceHolder', # For FA3 activation quantization, placeholders.

    # QuaRot interface
    'QuaRotInterface', # For QuaRot.
    'QuaRotOnlineInterface', # For QuaRotOnline.

    # save interface
    'AscendV1SaveInterface', # For AscendV1 save.
]

from msmodelslim.app.analysis_service.pipeline_interface import PipelineInterface as AnalyzePipelineInterface
from msmodelslim.app.naive_quantization.model_info_interface import ModelInfoInterface
from msmodelslim.app.quant_service.modelslim_v0.pipeline_interface import \
    PipelineInterface as ModelSlimPipelineInterfaceV0
from msmodelslim.app.quant_service.multimodal_sd_v1.pipeline_interface import \
    MultimodalPipelineInterface as MultimodalSDPipelineInterface
from msmodelslim.app.quant_service.modelslim_v1.save.interface import AscendV1SaveInterface
from msmodelslim.model import IModel
from msmodelslim.core.runner.pipeline_interface import PipelineInterface as ModelSlimPipelineInterfaceV1
from msmodelslim.quant.processor.anti_outlier.smooth_quant.interface import SmoothQuantInterface
from msmodelslim.quant.processor.anti_outlier.iter_smooth.interface import IterSmoothInterface
from msmodelslim.quant.processor.anti_outlier.flex_smooth.interface import FlexSmoothQuantInterface
from msmodelslim.quant.processor.kv_smooth import KVSmoothFusedInterface
from msmodelslim.quant.processor.quant.fa3.interface import FA3QuantAdapterInterface, FA3QuantPlaceHolder
from msmodelslim.quant.processor.quarot.quarot_interface import QuaRotInterface, QuaRotOnlineInterface
