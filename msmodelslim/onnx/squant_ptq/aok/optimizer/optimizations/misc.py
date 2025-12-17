# -*- coding: utf-8 -*-
# Copyright Huawei Technologies Co., Ltd. 2023-2023. All rights reserved.
from msmodelslim.onnx.squant_ptq.aok.optimizer.optimizations import AbstractOptimization
from msmodelslim.onnx.squant_ptq.aok.optimizer.transformations.abstract_transformation import AbstractTransformation
from msmodelslim.onnx.squant_ptq.aok.optimizer.optimizations.single_transform_optimization import \
    SingleTransformOptimization
from msmodelslim.onnx.squant_ptq.aok.optimizer.transformations import misc


class ChangeResizeOptimization(SingleTransformOptimization):

    def __init__(self) -> None:
        super(ChangeResizeOptimization, self).__init__(misc.ChangeResizeTransformation)

    def get_simple_name(self) -> str:
        return 'ChangeResize'


class LayerNormOptimization(AbstractOptimization):

    def get_simple_name(self) -> str:
        return 'LayerNorm'

    def _setup_transformations(self, op_version: int) -> [AbstractTransformation]:
        return [
            misc.ReplaceSubAddTransformation(op_version),
            misc.ReplaceMulMulTransformation(op_version),
            misc.ReplaceMulSubTransformation(op_version),
            misc.ReplaceReciprocalTransformation(op_version),
        ]


class RemoveDoubleResizeOptimization(SingleTransformOptimization):
    """
        Transformation prepared for DeepLab
    """

    def __init__(self) -> None:
        super(RemoveDoubleResizeOptimization, self).__init__(misc.RemoveDoubleResizeTransformation)

    def get_simple_name(self) -> str:
        return 'RemoveDoubleResize'


class ReplaceMaxPoolBlockOptimizationV1(SingleTransformOptimization):

    def __init__(self) -> None:
        super(ReplaceMaxPoolBlockOptimizationV1, self).__init__(
            misc.ReplaceMaxPoolBlockTransformationV1
        )

    def get_simple_name(self) -> str:
        return 'ReplaceMaxPoolBlockV1'


class ReplaceMaxPoolBlockOptimizationV2(SingleTransformOptimization):

    def __init__(self) -> None:
        super(ReplaceMaxPoolBlockOptimizationV2, self).__init__(
            misc.ReplaceMaxPoolBlockTransformationV2
        )

    def get_simple_name(self) -> str:
        return 'ReplaceMaxPoolBlockV2'


class ReplaceReluOptimization(SingleTransformOptimization):

    def __init__(self) -> None:
        super(ReplaceReluOptimization, self).__init__(misc.ReplaceReluTransformation)

    def get_simple_name(self) -> str:
        return 'ReplaceRelu'


class ReplaceRelu6Optimization(SingleTransformOptimization):

    def __init__(self) -> None:
        super(ReplaceRelu6Optimization, self).__init__(misc.ReplaceRelu6Transformation)

    def get_simple_name(self) -> str:
        return 'ReplaceRelu6'


class ReplaceSigmoidOptimizationV1(SingleTransformOptimization):

    def __init__(self) -> None:
        super(ReplaceSigmoidOptimizationV1, self).__init__(misc.ReplaceSigmoidTransformationV1)

    def get_simple_name(self) -> str:
        return 'ReplaceSigmoidV1'


class ReplaceSoftmaxOptimizationV1(SingleTransformOptimization):

    def __init__(self) -> None:
        super(ReplaceSoftmaxOptimizationV1, self).__init__(misc.ReplaceSoftmaxTransformationV1)

    def get_simple_name(self) -> str:
        return 'ReplaceSoftmaxV1'

    def _setup_transformations(self, op_version: int) -> [AbstractTransformation]:
        return [misc.ReplaceSoftmaxTransformationV1(mode='exp', op_version=op_version)]


class ReplaceSoftmaxOptimizationV2(SingleTransformOptimization):

    def __init__(self) -> None:
        super(ReplaceSoftmaxOptimizationV2, self).__init__(misc.ReplaceSoftmaxTransformationV2)

    def get_simple_name(self) -> str:
        return 'ReplaceSoftmaxV2'

    def _setup_transformations(self, op_version: int) -> [AbstractTransformation]:
        return [misc.ReplaceSoftmaxTransformationV2(mode='exp', op_version=op_version)]
