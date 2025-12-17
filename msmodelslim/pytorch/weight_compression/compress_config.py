# -*- coding: utf-8 -*-
# Copyright Huawei Technologies Co., Ltd. 2023-2023. All rights reserved.
from multiprocessing import cpu_count

from ascend_utils.common.security.type import check_type, check_element_type, check_character

from msmodelslim import logger


class CompressConfig:
    """ The configuration for LLM weight compression """
    
    def __init__(self, 
            do_pseudo_sparse=False, 
            sparse_ratio=1, 
            is_debug=False, 
            compress_disable_layers=None, 
            record_detail_root='./',
            multiprocess_num=1) -> object:
        """
            do_pseudo_sparse: whether to do pseudo sparse before compression
            sparse_ratio: percentile of non-zero values after pseudo sparse
            is_debug: print the compression ratio for each weight if is_debug is True
            compress_disable_layers: the layers in compress_disable_layers will 
                      not be compressed and directly saved in compress_output
            record_detail_root: the save path for the temporary data
        """
        self.do_pseudo_sparse = do_pseudo_sparse 
        self.sparse_ratio = sparse_ratio
        self.is_debug = is_debug 
        self.compress_disable_layers = [] if compress_disable_layers is None else compress_disable_layers 
        self.record_detail_root = record_detail_root
        self.logger = logger
        self.multiprocess_num = multiprocess_num

        self._check_params() 

    def _check_params(self, ):
        check_type(self.do_pseudo_sparse, bool, param_name="do_pseudo_sparse")
 
        if self.do_pseudo_sparse:
            check_type(self.sparse_ratio, (int, float), param_name="sparse_ratio")

        if self.do_pseudo_sparse and (self.sparse_ratio > 1.0 or self.sparse_ratio < 0.0):
            raise ValueError("sparse_ratio is invalid, expected an value in [0, 1], but got {}".format(
                                self.sparse_ratio)) 

        check_type(self.is_debug, bool, param_name="is_debug") 
        
        check_element_type(self.compress_disable_layers, str, (tuple, list), param_name="compress_disable_layers")
        check_character(self.compress_disable_layers, param_name="compress_disable_layers")

        if not isinstance(self.record_detail_root, str):
            raise ValueError('record_detail_root is invalid, expected a string, but got {}'.format(
                               self.record_detail_root))

        check_type(self.multiprocess_num, int, param_name="multiprocess_num")
        if self.multiprocess_num <= 0:
            raise ValueError('multiprocess_num is invalid, expected an int (>=1), but got {}'.format(
                               self.multiprocess_num))
        
        if self.multiprocess_num > cpu_count():
            self.multiprocess_num = 1
            self.logger.info(f"multiprocess_num: {self.multiprocess_num}, which is bigger than cpu_count \
                               of this machine ({cpu_count()}). Default multiprocess_num (1) will be set.")