# Copyright Huawei Technologies Co., Ltd. 2025. All rights reserved.
from typing import Generator, List, Optional, Literal, Annotated

from pydantic import Field, model_validator, AfterValidator

from msmodelslim.core.practice import PracticeConfig, Metadata
from msmodelslim.core.quant_service.modelslim_v1.quant_config import ModelslimV1ServiceConfig
from msmodelslim.core.quant_service.modelslim_v1.save import AscendV1Config
from msmodelslim.core.tune_strategy import ITuningStrategy
from msmodelslim.core.tune_strategy.base import BaseTuningStrategy
from msmodelslim.core.tune_strategy.dataset_loader_infra import DatasetLoaderInfra
from msmodelslim.core.tune_strategy.interface import StrategyConfig, EvaluateResult
from msmodelslim.core.tune_strategy.standing_high.standing_high_interface import StandingHighInterface
from msmodelslim.ir.qal import QScope, QDType
from msmodelslim.core.const import DeviceType
from msmodelslim.model import IModel
from msmodelslim.pytorch.llm_ptq.llm_ptq_tools.layer_select import LayerSelector
from msmodelslim.processor.base import AutoProcessorConfigList
from msmodelslim.processor.quant.linear import LinearProcessorConfig
from msmodelslim.core.quantizer.base import QConfig
from msmodelslim.core.quantizer.linear import LinearQConfig
from msmodelslim.utils.exception import SchemaValidateError, UnsupportedError
from msmodelslim.utils.logging import logger_setter, get_logger
from msmodelslim.utils.validation.pydantic import at_least_one_element


def _create_default_template() -> ModelslimV1ServiceConfig:
    """创建默认的V1模板PracticeConfig"""
    return ModelslimV1ServiceConfig(
        process=[
            LinearProcessorConfig(
                type="linear_quant",
                qconfig=LinearQConfig(
                    act=QConfig(
                        scope=QScope.PER_TENSOR,
                        dtype=QDType.INT8,
                        symmetric=False,
                        method="minmax"
                    ),
                    weight=QConfig(
                        scope=QScope.PER_CHANNEL,
                        dtype=QDType.INT8,
                        symmetric=True,
                        method="minmax"
                    )
                ),
                include=["*"],
                exclude=[]
            ),
        ],
        save=[
            AscendV1Config(type="ascendv1_saver", part_file_size=4),
        ],
        dataset="mix_calib.jsonl"
    )


def _get_default_metadata():
    return Metadata(
        config_id='standing_high',
        label={'w_bit': 8, 'a_bit': 8, 'is_sparse': False, 'kv_cache': False},
    )


class StandingHighStrategyConfig(StrategyConfig):
    """摸高算法策略配置（V1框架）"""
    type: Literal["standing_high"] = "standing_high"

    anti_outlier_strategies: Annotated[List[AutoProcessorConfigList], AfterValidator(at_least_one_element)]

    template: ModelslimV1ServiceConfig = Field(
        default_factory=_create_default_template,
        description="完整的PracticeConfig模板，用于提取所有配置（包括线性层量化）。如果未提供，将使用默认的V1模板"
    )

    metadata: Metadata = Field(default_factory=_get_default_metadata)

    @model_validator(mode='after')
    def validate_template_has_linear_quant(self):
        """校验template中必须有且只有一个linear_quant配置"""
        linear_quant_count = sum(
            1 for proc in self.template.process if isinstance(proc, LinearProcessorConfig))
        if linear_quant_count != 1:
            raise SchemaValidateError(f"template_practice must contain exactly one linear_quant processor, "
                                      f"found {linear_quant_count}")
        return self


@logger_setter("msmodelslim.core.tune_strategy.standing_high") 
class StandingHighStrategy(BaseTuningStrategy, ITuningStrategy):
    def __init__(self, config: StrategyConfig, dataset_loader: DatasetLoaderInfra):
        super().__init__(config, dataset_loader)

        if not isinstance(config, StandingHighStrategyConfig):
            raise SchemaValidateError(f"StandingHighStrategy requires StandingHighStrategyConfig, got {type(config)}")

        self.config: StandingHighStrategyConfig = config
        self.__counter = 0

    def generate_practice(self,
                          model: IModel,
                          device: DeviceType = DeviceType.NPU,
                          ) -> Generator[PracticeConfig, Optional[EvaluateResult], None]:
        """生成实践配置（V1框架）"""
        if not isinstance(model, StandingHighInterface):
            raise UnsupportedError(f"model must be StandingHighInterface, got {type(model)}")

        # 每次生成时重置计数器
        self.__counter = 0

        loaded_model = model.load_model(device=device)
        layer_selector = LayerSelector(model=loaded_model)

        dataset = self.dataset_loader.get_dataset_by_name(self.config.template.dataset)
        calib_tokens = model.handle_dataset(dataset, device=device)
        layer_selector.run(calib_tokens)

        # 释放显存
        loaded_model.to('meta')

        zero_evaluation = yield self._build_practice_config(self.config.anti_outlier_strategies[0], [])

        if zero_evaluation.is_satisfied:
            get_logger().info("Practice without disable layers satisfies demand.")
            return

        init_disable_level = yield from self._find_satisfied_disable_level(layer_selector)
        yield from self._stand_high(init_disable_level, layer_selector)

    def _build_practice_config(
            self,
            anti_outlier_processors: AutoProcessorConfigList,
            linear_quant_exclude: List[str],
    ) -> PracticeConfig:
        """构建PracticeConfig：离群值抑制processors放在最前面，然后拼接模板中的所有processors"""
        process: AutoProcessorConfigList = list(anti_outlier_processors)
        template = self.config.template

        for proc in template.process:
            if isinstance(proc, LinearProcessorConfig):
                process.append(LinearProcessorConfig(type="linear_quant", qconfig=proc.qconfig, include=proc.include,
                                                     exclude=linear_quant_exclude))
            else:
                process.append(proc)

        # 生成新的 config_id
        new_config_id = f"{self.config.metadata.config_id}_{self.__counter}"
        self.__counter += 1

        return PracticeConfig(
            apiversion="modelslim_v1",
            spec=ModelslimV1ServiceConfig(
                runner=template.runner,
                process=process,
                save=template.save,
                dataset=template.dataset
            ),
            metadata=Metadata(
                config_id=new_config_id,
                label=self.config.metadata.label,
            ),
        )

    def _find_satisfied_disable_level(
            self,
            layer_selector: LayerSelector,
    ) -> Generator[PracticeConfig, Optional[EvaluateResult], int]:
        """二分搜索找到满足需求的最小disable level（zero_evaluation一定不满足，所以从level 1开始）"""
        min_level, max_level = 1, len(layer_selector.layer_groups)

        while min_level < max_level:
            mid = (min_level + max_level) // 2
            exclude_names = layer_selector.select_layers_by_disable_level(mid)
            get_logger().debug("Trying disable level: %r, exclude: %r", mid, exclude_names)

            evaluation: EvaluateResult = yield self._build_practice_config(
                self.config.anti_outlier_strategies[0],
                exclude_names)

            if evaluation.is_satisfied:
                max_level = mid
            else:
                min_level = mid + 1

        get_logger().info("Fundamental disable level: %r", min_level)
        return min_level

    def _permute(self) -> Generator[AutoProcessorConfigList, None, None]:
        """排列离群值抑制策略，具有记忆功能：从上次成功的策略开始遍历"""
        if not hasattr(self, '_current_index'):
            self._current_index = 0

        choices = self.config.anti_outlier_strategies
        for _ in range(len(choices)):
            yield choices[self._current_index]
            self._current_index = (self._current_index + 1) % len(choices)

    def _stand_high(
            self,
            init_disable_level: int,
            layer_selector: LayerSelector
    ) -> Generator[PracticeConfig, Optional[EvaluateResult], None]:
        """执行摸高算法：逐步减少disable level，尝试不同的离群值抑制策略"""
        # 获得最小回退层级的量化配置
        anti_outlier_processors = self.config.anti_outlier_strategies[0]
        exclude_names = layer_selector.select_layers_by_disable_level(init_disable_level)
        best_practice_config = self._build_practice_config(anti_outlier_processors, exclude_names)

        # 初始化步长和当前回退级别
        reduce_level = 1
        current_disable_level = init_disable_level

        while current_disable_level - reduce_level >= 0:
            exclude_names = layer_selector.select_layers_by_disable_level(current_disable_level - reduce_level)
            get_logger().info("Current disable level: %r, try to reduce: %r",
                              current_disable_level, reduce_level)

            for anti_outlier_processors in self._permute():
                get_logger().debug("Trying config: exclude=%r, anti_outlier=%r", exclude_names,
                                   [p.type for p in anti_outlier_processors])
                practice = self._build_practice_config(anti_outlier_processors, exclude_names)
                evaluation: EvaluateResult = yield practice

                if evaluation.is_satisfied:
                    best_practice_config = practice
                    current_disable_level -= reduce_level
                    reduce_level = max(min(reduce_level * 2, current_disable_level), 1)
                    break
            else:
                if reduce_level == 1:
                    yield best_practice_config
                    return
                reduce_level = 1
                get_logger().info("Reset reduce level to 1")
            
        yield best_practice_config

        get_logger().info("No disable layer can satisfy demand")
