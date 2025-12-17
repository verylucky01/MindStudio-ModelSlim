# Copyright Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
from abc import ABC, abstractmethod


class DatasetProcessorBase(ABC):
    def __init__(self, dataset_path, tokenizer=None, model=None):
        self.dataset_path = dataset_path
        self.ori_prompts = []
        self.ori_answers = []
        self.tokenizer = tokenizer
        self.model = model

    @abstractmethod
    def process_data(self, indexs):
        """解析一组样本的数据格式"""
        prpt_ans = {}
        return prpt_ans

    @abstractmethod
    def verify_positive_prompt(self, prompts, labels):
        """校验一组样本是否为正样本"""
        prpt_ans = []
        return prpt_ans

    def get_dataset_size(self):
        return len(self.ori_prompts)