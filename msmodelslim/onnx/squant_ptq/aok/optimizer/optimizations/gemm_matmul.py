# -*- coding: utf-8 -*-
# Copyright Huawei Technologies Co., Ltd. 2023-2023. All rights reserved.
from msmodelslim.onnx.squant_ptq.aok.optimizer.optimizations import AbstractOptimization
from msmodelslim.onnx.squant_ptq.aok.optimizer.transformations.abstract_transformation import AbstractTransformation
from msmodelslim.onnx.squant_ptq.aok.optimizer.optimizations.single_transform_optimization import \
    SingleTransformOptimization
from msmodelslim.onnx.squant_ptq.aok.optimizer.transformations import gemm_matmul as gemm
from msmodelslim.onnx.squant_ptq.aok.optimizer.transformations import reshape


class CombineMatmulOptimization(SingleTransformOptimization):

    def __init__(self):
        super(CombineMatmulOptimization, self).__init__(gemm.CombineMatmulTransformation)

    def get_simple_name(self) -> str:
        return 'CombineMatmul'


class FuseDivMatmulOptimization(SingleTransformOptimization):

    def __init__(self):
        super(FuseDivMatmulOptimization, self).__init__(gemm.FuseDivMatmulTransformation)

    def get_simple_name(self) -> str:
        return 'FuseDivMatmul'


class Matmul2GemmOptimization(AbstractOptimization):

    def get_simple_name(self) -> str:
        return 'Matmul2Gemm'

    def _setup_transformations(self, op_version: int) -> [AbstractTransformation]:
        return [
            gemm.Matmul2GemmTransformation(op_version),
            reshape.DoubleReshapeTransformation(op_version),
        ]
