# Copyright Huawei Technologies Co., Ltd. 2023-2023. All rights reserved.
from torch import nn as nn

from msmodelslim.pytorch.sparse.sparse_tools.sparse_kia.wrapper import SparseModelWrapper
from msmodelslim.pytorch.sparse.sparse_tools.common.abstract_class import CompressModelWrapper
from msmodelslim.pytorch.sparse.sparse_tools.common.config import Config


class ModelWrapperFactory(object):
    @staticmethod
    def create_model_wrapper(wrapper_type: str,
                             model: nn.Module,
                             cfg: Config = None,
                             logger=None,
                             **kwargs) -> CompressModelWrapper:
        if wrapper_type.lower() == 'sparse':
            return SparseModelWrapper(model,
                                      cfg=cfg,
                                      logger=logger,
                                      dataset=kwargs.get("dataset"))
        else:
            raise NotImplementedError(f'{wrapper_type} wrapper is not supported')
