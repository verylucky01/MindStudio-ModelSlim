# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
import datetime
from copy import deepcopy
from pathlib import Path
from typing import Optional, Union, List

from msmodelslim.app.quant_service import IQuantService
from msmodelslim.app.tune_strategy import ITuningStrategyFactory
from msmodelslim.core.const import DeviceType
from msmodelslim.model import IModelFactory, IModel
from msmodelslim.utils.logging import logger_setter, get_logger
from msmodelslim.utils.validation.conversion import convert_to_readable_dir, convert_to_writable_dir, \
    convert_to_timedelta, \
    convert_to_bool
from msmodelslim.utils.validation.type import check_element_type, check_type
from .evaluation_service_infra import EvaluateServiceInfra, EvaluateContext
from .model_info_interface import ModelInfoInterface
from .plan_manager_infra import TuningPlanManagerInfra
from .practice_history_infra import TuningHistory, TuningHistoryManagerInfra
from .practice_manager_infra import PracticeManagerInfra

MAX_ITERATION = 30


@logger_setter()
class AutoTuningApplication:
    def __init__(self,
                 plan_manager: TuningPlanManagerInfra,
                 practice_manager: PracticeManagerInfra,
                 evaluation_service: EvaluateServiceInfra,
                 tuning_history_manager: TuningHistoryManagerInfra,
                 quantization_service: IQuantService,
                 model_factory: IModelFactory,
                 strategy_factory: ITuningStrategyFactory,
                 ) -> None:
        self.plan_manager = plan_manager
        self.practice_manager = practice_manager
        self.evaluation_service = evaluation_service
        self.quantization_service = quantization_service
        self.model_factory = model_factory
        self.strategy_factory = strategy_factory
        self.tuning_history_manager = tuning_history_manager

    def tune(self,
             model_type: str,
             model_path: Union[Path, str],
             save_path: Union[Path, str],
             plan_id: str,
             device: DeviceType = DeviceType.NPU,
             device_indices: Optional[List[int]] = None,
             timeout: Optional[Union[datetime.timedelta, str]] = None,
             trust_remote_code: bool = False) -> None:
        """
        Run the auto tuning application.
        Args:
            model_type: str, the type of the model
            model_path: Union[Path, str], the path of the model
            save_path: Union[Path, str], the directory to save the model
            plan_id: str, the id of the tuning plan
            device: DeviceType, the device to run the model
            device_indices: Optional[List[int]], the indices of the devices to run the model
            timeout: Optional[Union[datetime.timedelta, str]], the timeout of the tuning
            trust_remote_code: bool, whether to trust the remote code
        """
        check_type(model_type, str, param_name="model_type")
        model_path = convert_to_readable_dir(model_path, param_name="model_path")
        save_path = convert_to_writable_dir(save_path, param_name="save_path")
        check_type(plan_id, str, param_name="plan_id")
        check_type(device, DeviceType, param_name="device")
        if device_indices is not None:
            check_element_type(device_indices, int, param_name="device_indices")
        if timeout is not None:
            timeout = convert_to_timedelta(timeout, param_name="timeout")
        trust_remote_code = convert_to_bool(trust_remote_code, param_name="trust_remote_code")

        get_logger().info('Auto tuning with following parameters:')
        get_logger().info("model_type: %r", model_type)
        get_logger().info("model_path: %s", model_path)
        get_logger().info("save_path: %s", save_path)
        get_logger().info("plan_id: %r", plan_id)
        get_logger().info("device: %r", device)
        if device_indices:
            get_logger().info("device_indices: %r", device_indices)
        if timeout:
            get_logger().info("timeout: %r", timeout)
        get_logger().info("trust_remote_code: %r", trust_remote_code)

        self._tune(model_type, model_path, save_path, plan_id, device, device_indices, timeout,
                   trust_remote_code)

    def _tune(self,
              model_type: str,
              model_path: Path,
              save_path: Path,
              plan_id: str,
              device: DeviceType,
              device_indices: Optional[List[int]],
              timeout: datetime.timedelta,
              trust_remote_code: bool) -> None:
        # analyse model
        get_logger().info("===========ANALYSE MODEL===========")
        model_adapter = self.model_factory.create(model_type, model_path, trust_remote_code)
        get_logger().info("Using model adapter %r.", model_adapter.__class__.__name__)

        # create plan
        get_logger().info("===========CREATE TUNING PLAN===========")
        plan = self.plan_manager.get_plan_by_id(plan_id)
        get_logger().info("Using plan %r.", plan_id)

        # create strategy
        get_logger().info("===========CREATE TUNING STRATEGY===========")
        strategy = self.strategy_factory.create_strategy(strategy_config=plan.strategy)
        get_logger().info("Using strategy %r.", plan.strategy.type)

        # start tuning
        get_logger().info("===========START TUNING===========")
        datetime_start = datetime.datetime.now()
        allowed_end_time = datetime_start + timeout if timeout is not None else None
        get_logger().info("Start time: %s, timeout: %s", datetime_start, timeout)

        # create quant config generator
        practice_generator = strategy.generate_practice(model=model_adapter)
        evaluate_result = None
        practice = None
        # start tuning
        for count in range(MAX_ITERATION):
            # check timeout
            current_time = datetime.datetime.now()
            if allowed_end_time and current_time > allowed_end_time:
                get_logger().warning("Current time: %s exceed allowed end time: %s!",
                                     current_time, allowed_end_time)
                get_logger().warning("===========TIMEOUT===========")
                break

            try:
                get_logger().info("===========TRY %r===========", count)
                get_logger().info("Current time: %s", current_time)
                # generate quant config
                practice = practice_generator.send(evaluate_result)
                get_logger().debug("Practice: %r", practice)
                get_logger().info("Generate practice success")

                quant_model_path = save_path / f"quant_model"

                # quantize model
                self.quantization_service.quantize(
                    practice.model_copy(deep=True),
                    model_adapter=model_adapter,
                    save_path=quant_model_path,
                    device=device,
                    device_indices=device_indices
                )
                get_logger().info("Quantize model success")

                # evaluate model
                evaluate_result = self.evaluation_service.evaluate(
                    context=EvaluateContext(
                        evaluate_id=str(count),
                        device=device,
                        device_indices=device_indices,
                        working_dir=save_path,
                    ),
                    evaluate_config=plan.evaluation,
                    model_path=quant_model_path,
                )
                get_logger().info("Evaluate model success")
                for accuracy_unit in evaluate_result.accuracies:
                    get_logger().info("Evaluate Accuracy of %r: %r",
                                      accuracy_unit.dataset, accuracy_unit.accuracy)

                # save history
                history = TuningHistory(
                    practice=practice,
                    evaluation=evaluate_result,
                )
                self.tuning_history_manager.append_history(str(save_path / "history"), history)
                get_logger().info("Save history success")
            except StopIteration:
                get_logger().info("Strategy stop iterating")
                self._save_practice_to_custom_repo(model_adapter, practice)
                get_logger().info("===========SUCCESS===========")
                break
        else:
            get_logger().warning("===========EXCEED MAX TUNING ITERATION: %r===========", MAX_ITERATION)

    def _save_practice_to_custom_repo(self, model_adapter: IModel, practice):
        if not self.practice_manager.is_saving_supported():
            get_logger().warning(
                "Custom Practice Manager is not provided. "
                "Final practice will not be saved to Best Practice Repository.")
            return
        if not isinstance(model_adapter, ModelInfoInterface):
            get_logger().warning(
                "Model adapter %r does NOT implement ModelInfoInterface. "
                "Final practice will not be saved to Best Practice Repository",
                {model_adapter.__class__.__name__})
            return

        self.practice_manager.save_practice(
            model_pedigree=model_adapter.get_model_pedigree(),
            practice=practice)
        get_logger().info("Save practice to repo success")
