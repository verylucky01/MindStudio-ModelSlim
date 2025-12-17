# Copyright (c) 2023-2024 Huawei Technologies Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import random


def random_change_dict_value(ori_dict, random_value):
    '''
    Recursively modify a random value of dict
    '''
    key_list = [key for key in ori_dict.keys()]
    change_key = random.choice(key_list)
    change_value = ori_dict[change_key]
    if isinstance(change_value, dict):
        random_change_dict_value(change_value, random_value)
    elif isinstance(change_value, list) and change_value:
        change_index = random.randint(0, len(change_value) - 1)
        if isinstance(change_value[change_index], dict):
            random_change_dict_value(change_value[change_index], random_value)
        else:
            change_value[change_index] = random_value
    else:
        ori_dict[change_key] = random_value