# -*- coding: utf-8 -*-
# Copyright Huawei Technologies Co., Ltd. 2023-2023. All rights reserved.
import os
import numpy as np
from modelslim.pytorch.weight_compression import CompressConfig, Compressor
from msmodelslim import logger as msmodelslim_logger

def make_dir(path):
    if not os.path.exists(path):
        os.makedirs(path, mode=0o750)
    return path


def main(root):
    weight_path = os.path.join(root, "quant_weight.npy")
    save_path = f"{os.environ['PROJECT_PATH']}/output/weight_compression"
    index_root = make_dir(os.path.join(save_path, 'index'))
    weight_root = make_dir(os.path.join(save_path, 'weight'))
    info_root = make_dir(os.path.join(save_path, 'info'))

    config = CompressConfig(do_pseudo_sparse=False, sparse_ratio=1, is_debug=True,
                            record_detail_root=save_path, multiprocess_num=2)
    compressor = Compressor(config, weight_path)
    compress_weight, compress_index, compress_info = compressor.run()

    compressor.export(compress_weight, weight_root)
    compressor.export(compress_index, index_root)
    compressor.export(compress_info, info_root, dtype=np.int64)

src_root = f"{os.environ['PROJECT_PATH']}/resource/weight_compression"
main(src_root)
msmodelslim_logger.info("Weight Compression success!")