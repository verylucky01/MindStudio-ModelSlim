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
import shutil
from typing import Dict

import torch
from safetensors.torch import save_file

from msmodelslim.utils.security import get_valid_write_path, SafeWriteUmask, get_write_directory, \
    json_safe_dump

ONE_GB_FILE_BYTES = 1073741824  # 1G, 1 * 1024 * 1024 * 1024
FILE_TMP_SUFFIX = '-of-00000.safetensors'


def get_index_json(file_map_dict, total_size):
    index_json_dict = {
        'metadata': {'total_size': total_size},
        'weight_map': file_map_dict
    }
    return index_json_dict


class SafetensorsWriter:
    def __init__(self, logger, file_path):
        self.logger = logger
        file_path = get_valid_write_path(file_path, extensions=['safetensors'])
        self.file_path = file_path
        self.safetensors_weight = {}

    def write(self, key: str, value: torch.Tensor):
        self.safetensors_weight[key] = value.cpu().contiguous()

    def close(self):
        with SafeWriteUmask(umask=0o377):
            save_file(self.safetensors_weight, self.file_path)
        self.logger.info(f'Save safetensors to {self.file_path} successfully')


class BufferedSafetensorsWriter(SafetensorsWriter):

    def __init__(
            self,
            logger,
            max_gb_size: int = 4,
            save_directory: str = '.',
            save_prefix: str = "quant_model_weight"
    ):
        super().__init__(logger, save_prefix + '.safetensors')
        self.logger = logger
        self.saved_keys_map: Dict[str, str] = {}
        self.wait_save_keys: Dict[str, torch.Tensor] = {}
        self.max_size: int = max_gb_size * ONE_GB_FILE_BYTES
        self.save_prefix: str = save_prefix
        self.total_size: int = 0
        self._wait_save_size: int = 0
        self._save_count: int = 0
        self.save_directory: str = save_directory

    @property
    def save_directory(self) -> str:
        return self._save_directory

    @save_directory.setter
    def save_directory(self, value: str) -> None:
        self._save_directory = get_write_directory(value, write_mode=0o750)

    def save_index(self):
        # process safetensors index json
        if self._save_count <= 99999:
            suffix = f"-of-{self._save_count:05d}.safetensors"
        else:
            suffix = f"-of-{self._save_count}.safetensors"
        for key in self.saved_keys_map.keys():
            self.saved_keys_map[key] = self.saved_keys_map[key].replace(FILE_TMP_SUFFIX, suffix)

        # save index json
        index_json_dict = get_index_json(self.saved_keys_map, self.total_size)
        index_json_name = os.path.join(self.save_directory, self.save_prefix + '.safetensors.index.json')
        index_json_path = get_valid_write_path(index_json_name, extensions=[".json"])
        json_safe_dump(index_json_dict, index_json_path, indent=2)
        self.logger.debug(f'Save index json to {index_json_path} successfully')

    def save_one_file(self) -> None:
        # no tensors no saving
        if not self.wait_save_keys:
            return

        # one tensor larger than max size
        if self._wait_save_size > self.max_size:
            self.logger.warning(f'Tensor is too large with size {self._wait_save_size / ONE_GB_FILE_BYTES}GB, '
                                f'exceeds file size limit: {self.max_size / ONE_GB_FILE_BYTES}GB')

        self._save_count += 1
        save_file_name = f"{self.save_prefix}-{self._save_count:05d}{FILE_TMP_SUFFIX}"
        full_save_file_name = os.path.join(self.save_directory, save_file_name)
        full_save_file_name = get_valid_write_path(full_save_file_name, extensions=[".safetensors"])

        self.logger.debug(f"Start save {full_save_file_name}")
        with SafeWriteUmask(umask=0o377):
            save_file(self.wait_save_keys, full_save_file_name)
        self.saved_keys_map.update({key: save_file_name for key in self.wait_save_keys.keys()})
        self.wait_save_keys.clear()
        self._wait_save_size = 0
        self.logger.debug(f"End save {full_save_file_name}")

    def write(self, key: str, value: torch.Tensor) -> None:
        if value.device.type == 'meta':
            self.logger.warning(f"Skip meta tensor {key}")
            return

        tensor = value.detach().cpu().contiguous()

        tensor_size = tensor.numel() * tensor.element_size()
        if self._wait_save_size + tensor_size >= self.max_size:
            self.save_one_file()

        self.wait_save_keys[key] = tensor
        self.total_size += tensor_size
        self._wait_save_size += tensor_size

    def close(self) -> None:
        # save last file if needed
        self.save_one_file()

        # rename safetensors
        for i in range(self._save_count):
            src_file = os.path.join(self.save_directory, f"{self.save_prefix}-{i + 1:05d}{FILE_TMP_SUFFIX}")
            # 仿照开源权重命名均为model-0000x-of-0000x.safetensors，超过99999命名为model-x-of-x.safetensors
            if self._save_count <= 99999:
                dst_file_name = f"{self.save_prefix}-{i + 1:05d}-of-{self._save_count:05d}.safetensors"
            else:
                dst_file_name = f"{self.save_prefix}-{i + 1}-of-{self._save_count}.safetensors"
            dst_file = os.path.join(self.save_directory, dst_file_name)
            shutil.move(src_file, dst_file)
            self.logger.debug(f"{src_file} -> {dst_file}")
        self.logger.debug(f'Save .safetensors to {self.save_directory} successfully')

        self.save_index()
        self.logger.info(f'Save safetensors files to {self.save_directory} successfully')
