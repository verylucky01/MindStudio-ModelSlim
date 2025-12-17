# Copyright Huawei Technologies Co., Ltd. 2025. All rights reserved.
import torch
from safetensors.torch import save_file

from ascend_utils.common.security import get_valid_write_path, check_type, SafeWriteUmask
from msmodelslim.pytorch.llm_ptq.llm_ptq_tools.save.writer.base import BaseWriter


class SafetensorsWriter(BaseWriter):
    def __init__(self, logger, file_path):
        super().__init__(logger)

        file_path = get_valid_write_path(file_path, extensions=['safetensors'])
        self.file_path = file_path
        self.safetensors_weight = {}

    def _write(self, key: str, value: torch.Tensor):
        check_type(value, torch.Tensor)
        self.safetensors_weight[key] = value.cpu().contiguous()

    def _close(self):
        with SafeWriteUmask(umask=0o377):
            save_file(self.safetensors_weight, self.file_path)
        self.logger.info(f'Save safetensors to {self.file_path} successfully')