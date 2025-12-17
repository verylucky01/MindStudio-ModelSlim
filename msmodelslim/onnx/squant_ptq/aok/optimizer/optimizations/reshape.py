# -*- coding: utf-8 -*-
# Copyright Huawei Technologies Co., Ltd. 2023-2023. All rights reserved.
from msmodelslim.onnx.squant_ptq.aok.optimizer.optimizations.single_transform_optimization import \
    SingleTransformOptimization
from msmodelslim.onnx.squant_ptq.aok.optimizer.transformations import reshape


class DoubleReshapeOptimization(SingleTransformOptimization):

    def __init__(self):
        super(DoubleReshapeOptimization, self).__init__(reshape.DoubleReshapeTransformation)

    def get_simple_name(self) -> str:
        return 'DoubleReshape'


class SimplifyShapeOptimization(SingleTransformOptimization):

    def __init__(self):
        super(SimplifyShapeOptimization, self).__init__(reshape.SimplifyShapeTransformation)

    def get_simple_name(self) -> str:
        return 'SimplifyShape'
