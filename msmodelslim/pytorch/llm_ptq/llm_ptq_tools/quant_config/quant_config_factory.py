# Copyright Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.

from msmodelslim.pytorch.llm_ptq.llm_ptq_tools.quant_config.quant_config_classes import BaseConfig
from msmodelslim.pytorch.llm_ptq.llm_ptq_tools.quant_config.quant_config_classes import WeightQuantConfig
from msmodelslim.pytorch.llm_ptq.llm_ptq_tools.quant_config.quant_config_classes import WeightActivationQuantConfig
from msmodelslim.pytorch.llm_ptq.llm_ptq_tools.quant_config.quant_config_classes import SparseQuantConfig
from msmodelslim.pytorch.llm_ptq.llm_ptq_tools.quant_config.quant_config_classes import KVQuantConfig
from msmodelslim.pytorch.llm_ptq.llm_ptq_tools.quant_config.quant_config_classes import FAQuantConfig
from msmodelslim.pytorch.llm_ptq.llm_ptq_tools.quant_config.quant_config_classes import SimulateTPConfig
from msmodelslim.pytorch.llm_ptq.llm_ptq_tools.quant_config.quant_config_classes import TimestepQuantConfig


class QuantConfigFactory:
    """
    工厂模式，返回不同的量化Config类
    """
    QuantConfigMapper = {
        'base': BaseConfig,
        'weight': WeightQuantConfig,
        'weight_activation': WeightActivationQuantConfig,
        'sparse': SparseQuantConfig,
        'kv': KVQuantConfig,
        'fa_quant': FAQuantConfig,
        'simulate_tp': SimulateTPConfig,
        'timestep_quant': TimestepQuantConfig,
    }

    @classmethod
    def get_quant_config(cls, description: str, **kwargs) -> BaseConfig:
        if description in cls.QuantConfigMapper:
            return cls.QuantConfigMapper[description](**kwargs)
        raise ValueError(f"QuantConfig {description} does not support, please check your QuantConfig.")
