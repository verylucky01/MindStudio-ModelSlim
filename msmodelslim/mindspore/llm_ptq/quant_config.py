# Copyright Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.

from msmodelslim.pytorch.llm_ptq.llm_ptq_tools.quant_config.quant_config import QuantConfig as BaseQuantConfig


class QuantConfig(object):
    def __init__(self, 
                 disable_names: object = None,
                 fraction: float = 0.01):
        base_quant_cfg = BaseQuantConfig(
            disable_names=disable_names,
            fraction=fraction
        )
        config_attribute = base_quant_cfg.__dict__
        self.__dict__.update(**config_attribute)
