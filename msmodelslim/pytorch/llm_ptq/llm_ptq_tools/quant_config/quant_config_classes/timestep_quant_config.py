#  Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.

from ascend_utils.common.security import check_type, check_number
from msmodelslim.pytorch.llm_ptq.llm_ptq_tools.quant_config.quant_config_classes import BaseConfig
from msmodelslim.pytorch.llm_ptq.llm_ptq_tools.quant_config.quant_config_classes.config_utils import \
    check_and_generate_config_param


class TimestepQuantConfig(BaseConfig):
    """
    Timestep Quantization的Config
    """

    def __init__(self,
                 last_config: BaseConfig,
                 max_dynamic_step: int = None,
                 ):
        super().__init__()
        # 获取上一个config的所有属性，利用上一个config实例完成初始化，并进行参数的修改
        self._init_attribute_by_config(last_config)

        # 更新当前特有的属性
        self.use_timestep_quant = True
        self.max_dynamic_step = max_dynamic_step

        # 校验参数
        check_number(max_dynamic_step, int, min_value=0, param_name="max_dynamic_step")

        check_and_generate_config_param(self)
