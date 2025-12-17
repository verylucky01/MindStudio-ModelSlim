# Copyright Huawei Technologies Co., Ltd. 2025. All rights reserved.
from typing import List

from msmodelslim.pytorch.llm_ptq.llm_ptq_tools.save.saver.base import BaseSaver


class MultiSaver(BaseSaver):
    def __init__(self):
        super().__init__()

        self.saver_list: List[BaseSaver] = []

    def register(self, saver: BaseSaver):
        if not isinstance(saver, BaseSaver):
            raise TypeError(f'Saver must be a subclass of BaseSaver, not {type(saver).__name__}')
        self.saver_list.append(saver)

    def pre_process(self) -> None:
        for saver in self.saver_list:
            saver.pre_process()

    def save(self, name, meta, data) -> None:
        for saver in self.saver_list:
            saver.save(name, meta, data)

    def post_process(self) -> None:
        for saver in self.saver_list:
            saver.post_process()
