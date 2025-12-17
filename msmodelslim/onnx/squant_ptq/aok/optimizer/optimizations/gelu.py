# -*- coding: utf-8 -*-
# Copyright Huawei Technologies Co., Ltd. 2023-2023. All rights reserved.
from msmodelslim.onnx.squant_ptq.aok.optimizer.optimizations import AbstractOptimization
from msmodelslim.onnx.squant_ptq.aok.optimizer.transformations.abstract_transformation import AbstractTransformation
from msmodelslim.onnx.squant_ptq.aok.optimizer.optimizations.single_transform_optimization import \
    SingleTransformOptimization
from msmodelslim.onnx.squant_ptq.aok.optimizer.transformations import gelu


class FastClipOptimization(AbstractOptimization):

    def get_simple_name(self) -> str:
        return 'FastClip'

    def _setup_transformations(self, op_version: int) -> [AbstractTransformation]:
        return [
            gelu.ChangeClipTransformationV1(op_version=op_version),
            gelu.ChangeClipTransformationV2(op_version=op_version),
            gelu.ChangeClipTransformationV3(op_version=op_version)
        ]


class GeluErf2SigmoidOptimization(SingleTransformOptimization):

    def __init__(self):
        super(GeluErf2SigmoidOptimization, self).__init__(gelu.GeluErf2SigmoidTransformation)

    def get_simple_name(self) -> str:
        return 'GeluErf2Sigmoid'


class GeluTanh2SigmoidOptimization(SingleTransformOptimization):

    def __init__(self):
        super(GeluTanh2SigmoidOptimization, self).__init__(gelu.GeluTanh2SigmoidTransformation)

    def get_simple_name(self) -> str:
        return 'GeluTanh2Sigmoid'


class GeluErf2TanhOptimization(SingleTransformOptimization):

    def __init__(self):
        super(GeluErf2TanhOptimization, self).__init__(gelu.GeluErf2TanhTransformation)

    def get_simple_name(self) -> str:
        return 'GeluErf2Tanh'


class GeluErf2FastGeluOptimization(SingleTransformOptimization):

    def __init__(self):
        super(GeluErf2FastGeluOptimization, self).__init__(gelu.GeluErf2FastGeluTransformation)

    def get_simple_name(self) -> str:
        return 'GeluErf2FastGelu'


class ReplaceLeakyReluOptimization(SingleTransformOptimization):

    def __init__(self):
        super(ReplaceLeakyReluOptimization, self).__init__(gelu.ReplaceLeakyReluTransformation)

    def get_simple_name(self) -> str:
        return 'ReplaceLeakyRelu'
