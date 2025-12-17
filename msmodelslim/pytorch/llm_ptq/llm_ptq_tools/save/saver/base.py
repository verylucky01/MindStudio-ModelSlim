# Copyright Huawei Technologies Co., Ltd. 2025. All rights reserved.
from abc import abstractmethod, ABC


class BaseSaver(ABC):

    @abstractmethod
    def pre_process(self) -> None:
        pass

    @abstractmethod
    def save(self, name, meta, data) -> None:
        pass

    @abstractmethod
    def post_process(self) -> None:
        pass
