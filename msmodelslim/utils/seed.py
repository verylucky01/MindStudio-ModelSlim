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

import os
import random

import numpy as np
import torch
from packaging import version

from msmodelslim.utils.logging import get_logger

try:
    import torch_npu
except ImportError:
    is_gpu = True
else:
    is_gpu = False


def seed_all(seed=1234, mode=False):
    try:
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        cuda_version = torch.version.cuda
        if cuda_version is not None and version.parse(cuda_version) >= version.parse("10.2"):
            os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
        os.environ['HCCL_DETERMINISTIC'] = str(mode)
        torch.use_deterministic_algorithms(False)
        if is_gpu:
            torch.cuda.manual_seed_all(seed)
            torch.cuda.manual_seed(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.enable = False
            torch.backends.cudnn.benchmark = False
        else:
            torch_npu.npu.manual_seed_all(seed)
            torch_npu.npu.manual_seed(seed)
    except Exception as e:
        get_logger().error(f"There is an unexpected error while determinating randomness. {e}")
