# Copyright Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.

from msmodelslim.pytorch.llm_ptq.llm_ptq_tools.quant_config.quant_config_classes import BaseConfig
from msmodelslim.pytorch.llm_ptq.llm_ptq_tools.quant_config.quant_config_classes.config_utils import \
    check_and_generate_config_param


class WeightActivationQuantConfig(BaseConfig):
    """
    权重、激活量化的Config
    包含：W8A8，未来可能有W4A8
    """

    def __init__(self,
                 last_config: BaseConfig,
                 act_method: int = 1,
                 pr: float = 1.0,
                 is_dynamic: bool = False
                 ):
        """
        Args:
            last_config: 上一个config，基于该config进行修改
            act_method: activation量化的方式
            pr: 量化的一个概率参数，非1时量化生成的参数带有随机性
            is_dynamic: 是否使用动态量化，即w8a8中的activation动态生成
        """
        super().__init__()
        # 获取上一个config的所有属性
        self._init_attribute_by_config(last_config)

        # 更新当前特有的属性
        self.act_method = act_method  # 1为Min-Max，2为Histogram，3为1+2自动混合，建议用3
        self.pr = pr  # 一种随机参数，建议为1，不进行手工调整
        self.is_dynamic = is_dynamic  # 设置为动态量化，即per-token

        # 校验参数
        check_and_generate_config_param(self)
