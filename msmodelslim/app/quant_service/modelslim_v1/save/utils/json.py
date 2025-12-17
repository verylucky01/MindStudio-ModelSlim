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

from ascend_utils.common.security import json_safe_dump


class JsonWriter:
    def __init__(self, save_directory: str, file_name: str):
        self.save_directory = save_directory
        self.file_name = file_name
        self.value_map = {}

    def write(self, prefix: str, desc: object):
        self.value_map[prefix] = desc

    def close(self):
        json_safe_dump(self.value_map, os.path.join(self.save_directory, self.file_name), indent=4)
