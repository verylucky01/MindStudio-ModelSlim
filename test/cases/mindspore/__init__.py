# -*- coding: utf-8 -*-
# Copyright Huawei Technologies Co., Ltd. 2021-2021. All rights reserved.

# NOTE: even in MindSpore test cases, we must import torch in the first place
# or it will cause following import error when importing cv2 which needed by vega:
# ImportError: dlopen: cannot load any more object with static TLS
import torch
