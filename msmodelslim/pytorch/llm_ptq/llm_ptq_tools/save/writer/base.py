# Copyright Huawei Technologies Co., Ltd. 2025. All rights reserved.
from abc import ABC, abstractmethod


class BaseWriter(ABC):
    def __init__(self, logger):
        self.logger = logger
        self.is_closed = False

    def __setitem__(self, key, value):
        self.write(key, value)

    @abstractmethod
    def _write(self, key, value) -> None:
        ...

    @abstractmethod
    def _close(self) -> None:
        ...

    def write(self, key, value):
        if self.is_closed:
            return

        self._write(key, value)

    def close(self):
        if self.is_closed:
            return

        self.is_closed = True
        self._close()