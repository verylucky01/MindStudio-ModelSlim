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

__all__ = [
    "quant_model",
    "SessionConfig",
    "M3ProcessorConfig",
    "M4ProcessorConfig",
    "M6Config",
    "M6ProcessorConfig",
    "W8A8QuantConfig",
    "W8A8ProcessorConfig",
    "FA3ProcessorConfig",
    "W8A8DynamicQuantConfig",
    "W8A8DynamicProcessorConfig",
    "W8A8TimeStepQuantConfig",
    "W8A8TimeStepProcessorConfig",
    "SaveProcessorConfig",
]

try:
    from msmodelslim.quant.session.session import quant_model, SessionConfig
    from msmodelslim.quant.session.session import M4ProcessorConfig, W8A8QuantConfig, W8A8ProcessorConfig, \
        FA3ProcessorConfig, W8A8DynamicQuantConfig, W8A8DynamicProcessorConfig, W8A8TimeStepQuantConfig, \
        W8A8TimeStepProcessorConfig, SaveProcessorConfig, M3ProcessorConfig, M6ProcessorConfig, M6Config
except ImportError as e:
    from msmodelslim.utils.logging import get_logger

    get_logger().warning(
        f"The session module is imported failed, but it is not a critical error, because v1 does not use it")
