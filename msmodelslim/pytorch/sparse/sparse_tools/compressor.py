# Copyright Huawei Technologies Co., Ltd. 2023-2023. All rights reserved.
import torch.nn as nn

from ascend_utils.common.security import check_type, check_element_type, check_character
from msmodelslim.pytorch.sparse.sparse_tools.common.factory import ModelWrapperFactory
from msmodelslim.pytorch.sparse.sparse_tools.compression.sparse.sparse_config import SparseConfig
from msmodelslim import logger


class Compressor(object):
    """ Compressor to manage the compression process."""
    def __init__(self,
                 model: nn.Module,
                 cfg: SparseConfig):
        self.model = model
        self.cfg = cfg
        self.logger = logger
        self.model_wrapper = None
        self.is_compressed = False
        self._check_params()

    @property
    def compressed_model(self):
        if self.is_compressed:
            return self.model
        result = None
        return result

    def compress(self,
                 dataset):
        """ Wrap the model for compression."""
        check_element_type(dataset, str, list, param_name="dataset")
        check_character(dataset, param_name="dataset")
        
        try:
            self.model_wrapper = ModelWrapperFactory.create_model_wrapper(self.cfg.mode,
                                                                          self.model,
                                                                          cfg=self.cfg,
                                                                          logger=self.logger,
                                                                          dataset=dataset)
        except Exception as e:
            self.logger.error(e)
            raise

        self.model_wrapper.wrap()
        self.is_compressed = True
    
    def _check_params(self):
        check_type(self.model, nn.Module, param_name="model")
        check_type(self.cfg, SparseConfig, param_name="cfg")

