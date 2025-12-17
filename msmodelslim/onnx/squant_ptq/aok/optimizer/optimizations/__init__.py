# -*- coding: utf-8 -*-
# Copyright Huawei Technologies Co., Ltd. 2023-2023. All rights reserved.
from inspect import isclass
from msmodelslim.onnx.squant_ptq.aok.optimizer.optimizations.abstract_optimization import AbstractOptimization


def import_all_optimizations(file: str, name: str) -> None:
    from pkgutil import iter_modules
    from pathlib import Path
    from importlib import import_module

    _package_dir = str(Path(file).resolve().parent)
    for (_, module_name, _) in iter_modules([_package_dir]):
        module = import_module(f'{name}.{module_name}')
        for attribute_name in dir(module):
            attribute = getattr(module, attribute_name)
            if isclass(attribute) and issubclass(attribute, AbstractOptimization):
                globals()[attribute_name] = attribute


import_all_optimizations(__file__, __name__)
