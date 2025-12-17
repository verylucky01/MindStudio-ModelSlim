#  Copyright (c) Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.
from ascend_utils.common.security import check_type
from msmodelslim.pytorch.llm_ptq.llm_ptq_tools.quant_config.quant_config_classes import BaseConfig
from msmodelslim.pytorch.llm_ptq.llm_ptq_tools.quant_config.quant_config_classes.config_utils import \
    check_and_generate_config_param

SUPPORT_TP_SIZES = [2, 4, 8, 16]


class SimulateTPConfig(BaseConfig):
    """
    simulate tp的Config
    """

    def __init__(self,
                 last_config: BaseConfig,
                 tp_size,
                 enable_communication_quant=True,
                 enable_per_device_quant=False):
        super().__init__()
        # 获取上一个config的所有属性，利用上一个config实例完成初始化，并进行参数的修改
        self._init_attribute_by_config(last_config)

        # 更新当前特有的属性
        self.tp_size = tp_size
        self.enable_communication_quant = enable_communication_quant
        self.enable_per_device_quant = enable_per_device_quant
        self.simulate_bit = 8

        # 校验参数
        check_and_generate_config_param(self)
        if tp_size not in SUPPORT_TP_SIZES:
            raise ValueError(f'tp_size should be in f{SUPPORT_TP_SIZES}, but got {tp_size}.')
        check_type(enable_per_device_quant, bool, param_name='enable_per_device_quant')
        check_type(enable_communication_quant, bool, param_name='enable_communication_quant')
