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


from collections import OrderedDict
from collections.abc import Mapping
from typing import Any, Generic, TypeVar, Set, List

from wcmatch import fnmatch

T = TypeVar('T')


class ConfigMap(Generic[T], Mapping):
    def __init__(self, cfg_map: OrderedDict[str, T]):
        self.cfg_map: OrderedDict[str, Any] = cfg_map

    def __getitem__(self, key: str) -> T:
        if key in self.cfg_map:
            return self.cfg_map[key]
        for pattern in self.cfg_map:
            if fnmatch.fnmatch(key, pattern, flags=fnmatch.BRACE):
                return self.cfg_map[pattern]
        raise KeyError(f"Key '{key}' not found in config map")

    def __contains__(self, key: str) -> bool:
        if key in self.cfg_map:
            return True
        for pattern in self.cfg_map:
            if fnmatch.fnmatch(key, pattern, flags=fnmatch.BRACE):
                return True
        return False

    def __iter__(self):
        return iter(self.cfg_map)

    def __len__(self):
        return len(self.cfg_map)


class ConfigSet(Generic[T], Set):
    def __init__(self, cfg_list: List[T]):
        self.cfg_set = OrderedDict().fromkeys(cfg_list)
        self.matched_patterns = set()

    def __contains__(self, key: T) -> bool:
        if key in self.cfg_set:
            self.matched_patterns.add(key)
            return True
        for pattern in self.cfg_set:
            if fnmatch.fnmatch(key, pattern, flags=fnmatch.BRACE):
                self.matched_patterns.add(pattern)
                return True
        return False

    def __iter__(self):
        return iter(self.cfg_set)

    def __len__(self):
        return len(self.cfg_set)

    def unmatched_keys(self) -> Set[str]:
        unmatched_keys = set(self.cfg_set.keys()) - self.matched_patterns
        return unmatched_keys
