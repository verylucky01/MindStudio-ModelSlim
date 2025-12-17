#  Copyright (c) Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.

from ascend_utils.common.security import check_type, check_number
from msmodelslim.pytorch.llm_ptq.llm_ptq_tools.quant_config.quant_config_classes import BaseConfig
from msmodelslim.pytorch.llm_ptq.llm_ptq_tools.quant_config.quant_config_classes.config_utils import \
    check_and_generate_config_param


SUPPORT_TP_SIZES = [1, 2, 4, 8, 16]


class FAQuantConfig(BaseConfig):
    """
    FlashAttention Quantization的Config
    """

    def __init__(self,
                 last_config: BaseConfig,
                 fa_amp: int = 0
                 ):
        super().__init__()
        # 获取上一个config的所有属性，利用上一个config实例完成初始化，并进行参数的修改
        self._init_attribute_by_config(last_config)

        # 更新当前特有的属性
        self.use_fa_quant = True
        self.fa_tp_size = 1
        self.fa_amp = fa_amp

        # 校验参数
        if self.fa_tp_size not in SUPPORT_TP_SIZES:
            raise ValueError(f'fa_tp_size should be in f{SUPPORT_TP_SIZES}, but got {self.fa_tp_size}.')
        check_number(fa_amp, int, min_value=0, param_name="fa_amp")

        check_and_generate_config_param(self)
