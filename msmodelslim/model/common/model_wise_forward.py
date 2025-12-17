#  -*- coding: utf-8 -*-
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

from typing import List, Union, Tuple, Dict

import torch

from msmodelslim.core.base.protocol import ProcessRequest


def model_wise_forward_func(model: torch.nn.Module,
                            inputs: Union[List, Tuple, Dict],
                            ):
    yield ProcessRequest("", model,
                         inputs if isinstance(inputs, list) or isinstance(inputs, tuple) else [inputs],
                         inputs if isinstance(inputs, dict) else {})


def model_wise_visit_func(model: torch.nn.Module, ):
    yield ProcessRequest("", model, tuple(), {})
