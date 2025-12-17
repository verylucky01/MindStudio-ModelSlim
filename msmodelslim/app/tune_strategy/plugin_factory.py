# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
from msmodelslim.app.tune_strategy.interface import ITuningStrategyFactory, ITuningStrategy, StrategyConfig
from msmodelslim.utils.plugin.typed_factory import TypedFactory
from .dataset_loader_infra import DatasetLoaderInfra

TUNING_STRATEGY_PLUGIN_PATH = "msmodelslim.strategy.plugins"


class PluginTuningStrategyFactory(ITuningStrategyFactory):
    def __init__(self, dataset_loader: DatasetLoaderInfra):
        """
        初始化调优策略工厂
        
        使用 TypedFactory 来管理策略类的动态加载和实例化
        """
        self._factory = TypedFactory[ITuningStrategy](
            entry_point_group=TUNING_STRATEGY_PLUGIN_PATH,
            config_base_class=StrategyConfig,
        )
        self.dataset_loader = dataset_loader

    def create_strategy(self, strategy_config: StrategyConfig) -> ITuningStrategy:
        return self._factory.create(strategy_config, dataset_loader=self.dataset_loader)
