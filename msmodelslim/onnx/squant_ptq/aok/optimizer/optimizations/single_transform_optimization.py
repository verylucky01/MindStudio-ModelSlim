# -*- coding: utf-8 -*-
# Copyright Huawei Technologies Co., Ltd. 2023-2023. All rights reserved.
from abc import ABC, abstractmethod
from inspect import isabstract
from typing import Generic, TypeVar, Type, Union
from msmodelslim.onnx.squant_ptq.aok.optimizer.optimizations.abstract_optimization import AbstractOptimization
from msmodelslim.onnx.squant_ptq.aok.optimizer.transformations.abstract_transformation import AbstractTransformation

T = TypeVar('T', bound=AbstractTransformation)


class SingleTransformOptimization(AbstractOptimization, Generic[T], ABC):

    @abstractmethod
    def __init__(self, t: Union[T, Type[T]]):
        super(AbstractOptimization, self).__init__()
        if not isinstance(t, AbstractTransformation) and not issubclass(t, AbstractTransformation):
            raise TypeError(f't must be an instance or subclass of AbstractTransformation but got {type(t)}')
        self.__transform = t

    def get_simple_name(self):
        return self.__class__.__name__ if not isabstract(self.__class__) \
            else self.__transform.get_simple_name() if isinstance(self.__transform, AbstractTransformation) \
            else self.__transform.__name__ if issubclass(self.__transform, AbstractTransformation) \
            else None

    def _setup_transformations(self, op_version: int) -> [AbstractTransformation]:
        t = self.__transform if isinstance(self.__transform, AbstractTransformation) \
            else self.__transform(op_version) if issubclass(self.__transform, AbstractTransformation) \
            else None
        return [t]
