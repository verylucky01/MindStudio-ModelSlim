#  -*- coding: utf-8 -*-
#  Copyright (c) 2025-2025 Huawei Technologies Co., Ltd.
#  #
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#  #
#  http://www.apache.org/licenses/LICENSE-2.0
#  #
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

from typing import Dict, Any, List, Optional, Literal, Tuple, Union

import torch
from pydantic import BaseModel, Field, model_validator
from torch import nn

import msmodelslim.ir as qir
from msmodelslim.ir.qal import QParam, QStorage, QScope, QScheme, QABCRegistry, QDType
from msmodelslim.core.base.protocol import BatchProcessRequest
from msmodelslim.core.quantizer.base import QConfig
from msmodelslim.core.quantizer.linear import LinearQConfig
from msmodelslim.processor.base import AutoSessionProcessor, AutoProcessorConfig
from msmodelslim.utils.config_map import ConfigSet
from msmodelslim.utils.exception import SchemaValidateError
from msmodelslim.utils.logging import logger_setter, get_logger
from .autoround_utils.trainer import BlockQuantTrainer
from .autoround_utils.utils import get_shared_keys
from .autoround_utils.wrapper import WrapperLinear

# 常量定义
DEFAULT_BITS = 16
DEFAULT_ACT_BITS = 16
DEFAULT_GROUP_SIZE = -1
QUANTIZATION_THRESHOLD = 8

# 支持的层类型
SUPPORTED_LAYER_TYPES = (torch.nn.Linear,)

# 平滑层模式
SMOOTH_LAYER_PATTERNS = ['.down_proj', '.o_proj', 'kv_b_proj']


def _check_to_quantized(config: Union[Dict[str, Any], Any]) -> bool:
    """Checks if the configuration is valid for quantization.

    Args:
        config: The configuration to check. It can be either a
            dictionary with a 'bits' key or an object with a 'bits' attribute.

    Returns:
        True if the configuration is valid for quantization (bits <= 8),
        False otherwise.
    """

    def _get_bits(config_obj: Any, attr_name: str) -> int:
        """Helper function to extract bits from config object"""
        default_value = DEFAULT_BITS if attr_name == "bits" else DEFAULT_ACT_BITS
        if isinstance(config_obj, dict):
            return int(config_obj.get(attr_name, default_value))
        elif hasattr(config_obj, "orig_layer"):
            return int(getattr(config_obj.orig_layer, attr_name, default_value))
        else:
            return int(getattr(config_obj, attr_name, default_value))

    bits = _get_bits(config, "bits")
    act_bits = _get_bits(config, "act_bits")
    should_quantize = bits <= QUANTIZATION_THRESHOLD or act_bits <= QUANTIZATION_THRESHOLD

    get_logger().debug(f"Quantization check: bits={bits}, act_bits={act_bits}, should_quantize={should_quantize}")
    return should_quantize


def _mark_smooth_layers(module: nn.Module, name: str) -> None:
    """标记需要平滑的层"""
    if any(pattern in name for pattern in SMOOTH_LAYER_PATTERNS):
        module.to_smooth = True
        get_logger().debug(f"Marked layer '{name}' for smoothing")


def _merge_input_others(request_datas: List[Tuple[List[Any], Dict[str, Any]]]) -> Dict[str, Any]:
    """合并input_others数据
    
    Args:
        request_datas: 请求数据列表，每个元素包含input_ids和input_others
        
    Returns:
        合并后的input_others字典
    """
    merged_input_others = {}

    # 收集所有input_others
    for _, input_others in request_datas:
        for key, value in input_others.items():
            if key not in merged_input_others:
                merged_input_others[key] = []
            merged_input_others[key].append(value)

    # 处理特殊字段
    for key, value in merged_input_others.items():
        if isinstance(value, list) and not all(isinstance(v, torch.Tensor) for v in value):
            if key != "position_embeddings":
                merged_input_others[key] = value[0] if value else None

    return merged_input_others


def _extract_input_ids(request_datas: List[Tuple[List[Any], Dict[str, Any]]]) -> List[Any]:
    """提取所有input_ids
    
    Args:
        request_datas: 请求数据列表
        
    Returns:
        所有input_ids的列表
    """
    all_input_ids = []
    for input_ids, _ in request_datas:
        if isinstance(input_ids, list):
            all_input_ids.extend(input_ids[0])
        else:
            all_input_ids.append(input_ids[0])
    return all_input_ids


def _convert_request_datas_format(request_datas: List[Tuple[List[Any], Dict[str, Any]]]) -> Tuple[
    List[Any], Dict[str, Any]]:
    """将request.datas格式转换为训练器所需的格式
    
    Args:
        request_datas: 请求数据列表
        
    Returns:
        转换后的(input_ids, input_others)元组
        
    Raises:
        ValueError: 当request_datas为空时
    """
    if not request_datas:
        raise ValueError("request_datas不能为空")

    # 提取所有input_ids
    all_input_ids = _extract_input_ids(request_datas)

    # 合并input_others
    merged_input_others = _merge_input_others(request_datas)
    merged_input_others["positional_inputs"] = ()

    return all_input_ids, merged_input_others


def _apply_default_config(module: nn.Module) -> None:
    """应用默认配置
    
    Args:
        module: 要应用配置的模块
    """
    setattr(module, 'bits', DEFAULT_BITS)
    setattr(module, 'act_bits', DEFAULT_ACT_BITS)


def _apply_config_to_module(module: nn.Module, config: Dict[str, Any]) -> None:
    """将配置应用到模块
    
    Args:
        module: 要应用配置的模块
        config: 配置字典
    """
    for key, value in config.items():
        if key != 'qconfig':
            setattr(module, key, value)


def _wrapper_block(
        block: nn.Module,
        enable_minmax_tuning: bool,
        enable_round_tuning: bool,
        enable_trainable_smooth: bool = False,
        config: Optional[Dict[str, Any]] = None,
        **kwargs: Any
) -> Tuple[List[str], List[str]]:
    """包装块中的层为自定义Wrapper模块
    
    Args:
        block: 要包装的模块块
        enable_minmax_tuning: 是否启用最小最大值调优
        enable_round_tuning: 是否启用舍入调优
        enable_trainable_smooth: 是否启用可训练平滑
        config: 配置字典
        **kwargs: 其他参数
        
    Returns:
        量化层名称列表和未量化层名称列表的元组
    """
    quantized_layers = []
    unquantized_layers = []

    get_logger().debug(
        f"Starting to wrap block with {enable_minmax_tuning=}, {enable_round_tuning=}, {enable_trainable_smooth=}")

    for n, m in block.named_modules():
        if isinstance(m, SUPPORTED_LAYER_TYPES):
            _mark_smooth_layers(m, n)

            if not _check_to_quantized(m):
                unquantized_layers.append(n)
                get_logger().debug(f"Layer '{n}' skipped quantization")
                continue

            new_m = WrapperLinear(
                m,
                enable_minmax_tuning=enable_minmax_tuning,
                enable_round_tuning=enable_round_tuning,
                enable_trainable_smooth=enable_trainable_smooth,
                config=config,
                **kwargs,
            )

            block.set_submodule(n, new_m)
            quantized_layers.append(n)
            get_logger().debug(f"Layer '{n}' wrapped for quantization")

    get_logger().info(
        f"Block wrapping completed: {len(quantized_layers)} quantized, {len(unquantized_layers)} unquantized")
    return quantized_layers, unquantized_layers


def _parse_dtype(dtype: QDType, default_bits: int, default_type: str) -> Tuple[int, str]:
    """解析数据类型，返回bits和data_type
    
    Args:
        dtype: 数据类型
        default_bits: 默认位数
        default_type: 默认类型
        
    Returns:
        (bits, data_type)元组
    """
    dtype_name = dtype.name.lower()
    dtype_mapping = {
        'int8': (8, 'int'),
        'int4': (4, 'int'),
        'float': (DEFAULT_BITS, 'float'),
    }
    return dtype_mapping.get(dtype_name, (default_bits, default_type))


def _parse_scale_dtype(dtype_str: str) -> torch.dtype:
    """解析scale数据类型
    
    Args:
        dtype_str: 数据类型字符串
        
    Returns:
        对应的torch数据类型
        
    Raises:
        SchemaValidateError: 当数据类型不在支持范围内时抛出异常
    """
    support_scale_dtypes = {
        "float16": torch.float16,
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
    }

    if dtype_str not in support_scale_dtypes:
        raise SchemaValidateError(
            f"Unsupported scale dtype '{dtype_str}', supported types are: {list(support_scale_dtypes.keys())}",
            action=f"Please use one of the supported scale dtypes: {list(support_scale_dtypes.keys())}"
        )

    return support_scale_dtypes[dtype_str]


def _extract_activation_config(act_config: QConfig) -> Dict[str, Any]:
    """提取激活相关配置
    
    Args:
        act_config: 激活配置对象
        
    Returns:
        激活配置字典
    """
    bits, data_type = _parse_dtype(act_config.dtype, default_bits=DEFAULT_ACT_BITS, default_type='float')
    return {
        'act_bits': bits,
        'act_data_type': data_type,
        'act_sym': act_config.symmetric,
        'act_group_size': act_config.ext.get('group_size', DEFAULT_GROUP_SIZE),
        'act_dynamic': act_config.scope == QScope.PER_TOKEN,
    }


def _extract_weight_config(weight_config: Any) -> Dict[str, Any]:
    """提取权重相关配置
    
    Args:
        weight_config: 权重配置对象
        
    Returns:
        权重配置字典
    """
    bits, data_type = _parse_dtype(weight_config.dtype, default_bits=8, default_type='int')
    return {
        'bits': bits,
        'data_type': data_type,
        'sym': weight_config.symmetric,
        'group_size': weight_config.ext.get('group_size', DEFAULT_GROUP_SIZE),
        'scale_dtype': _parse_scale_dtype(weight_config.ext.get('scale_dtype', 'bfloat16')),
    }


def _create_layer_config(qconfig: LinearQConfig) -> Dict[str, Any]:
    """创建单层的配置
    
    Args:
        qconfig: 线性量化配置
        
    Returns:
        层配置字典
    """
    config = {
        'qconfig': qconfig,
        'weight_qconfig': qconfig.weight,
        'act_qconfig': qconfig.act,
        'super_bits': None,
        'super_group_size': None,
    }

    # 权重配置
    config.update(_extract_weight_config(qconfig.weight))

    # 激活配置
    config.update(_extract_activation_config(qconfig.act))

    return config


def _should_apply_strategy(
        layer_name: str,
        include_set: ConfigSet,
        exclude_set: ConfigSet,
        layer_config: Dict[str, Any]
) -> bool:
    """检查是否应该对层应用策略
    
    Args:
        layer_name: 层名称
        include_set: 包含集合
        exclude_set: 排除集合
        layer_config: 层配置字典
        
    Returns:
        是否应该应用策略
    """

    included = layer_name in include_set
    excluded = layer_name in exclude_set
    should_apply = included and not excluded

    if should_apply and layer_name in layer_config:
        original_config = layer_config[layer_name]
        get_logger().warning(
            f"Layer '{layer_name}' configuration already exists, "
            f"skipping to preserve original configuration: {original_config}")
        return False

    get_logger().debug(
        f"Strategy application check for '{layer_name}': "
        f"included={included}, excluded={excluded}, should_apply={should_apply}")
    return should_apply


def _warning_unmatched_pattern(name: str, config_set: ConfigSet) -> None:
    """警告未匹配的模式
    
    Args:
        name: 模式名称
        config_set: 配置集合
    """
    unmatched_keys = config_set.unmatched_keys()
    unmatched_keys = [key for key in unmatched_keys if key != "*"]
    if unmatched_keys:
        get_logger().warning(
            f"These {name} patterns are not matched any module, please ensure this is as expected: {unmatched_keys}")


def _create_fake_quantizer(orig_layer: torch.nn.Linear) -> qir.AutoFakeQuantLinear:
    """创建fake quantizer

    Args:
        orig_layer: 原始层

    Returns:
        fake quantizer实例
    """
    w_q_scheme: QScheme = orig_layer.weight_qconfig.to_scheme()
    group_size = orig_layer.weight_qconfig.ext.get("group_size", -1)
    scale = torch.tensor(orig_layer.scale).squeeze(1)
    offset = torch.tensor(orig_layer.zp).squeeze(1)

    get_logger().debug(
        f"Creating fake quantizer: group_size={group_size}, scale_shape={scale.shape}, offset_shape={offset.shape}")

    w_q_param = QParam(
        scheme=w_q_scheme,
        ext={
            "scale": scale,
            "offset": offset,
            "group_size": group_size
        }
    )
    x_q_scheme: QScheme = orig_layer.act_qconfig.to_scheme()
    x_q_param = QParam(scheme=x_q_scheme)
    w_q = QStorage(dtype=w_q_scheme.dtype, value=orig_layer.weight)

    get_logger().debug(f"Fake quantizer parameters: weight_scheme={w_q_scheme}, activation_scheme={x_q_scheme}")

    return qir.AutoFakeQuantLinear.create(
        x_q_param,
        w_q_param,
        w_q,
        orig_layer.bias
    )


class QuantStrategyConfig(BaseModel):
    qconfig: LinearQConfig = Field(description="量化配置")
    include: List[str] = Field(default_factory=lambda: ["*"], description="包含的模块名称")
    exclude: List[str] = Field(default_factory=list, description="排除的模块名称")


def _validate_quantization_strategies(strategies: List[QuantStrategyConfig]) -> None:
    """校验量化策略配置
    
    Args:
        strategies: 量化策略配置列表
        
    Raises:
        SchemaValidateError: 当校验失败时抛出异常
    """
    # 检查是否为空列表
    if not strategies:
        raise SchemaValidateError(
            "strategies field cannot be empty, at least one quantization strategy must be configured",
            action="Please add at least one QuantStrategyConfig to the strategies field"
        )

    for i, strategy in enumerate(strategies):
        qconfig = strategy.qconfig

        # 校验权重配置
        _validate_qconfig_group_size(
            qconfig.weight,
            f"strategies[{i}].qconfig.weight"
        )

        # 校验激活配置
        _validate_qconfig_group_size(
            qconfig.act,
            f"strategies[{i}].qconfig.act"
        )


def _validate_qconfig_group_size(qconfig: QConfig, field_path: str) -> None:
    """校验单个QConfig的group_size字段
    
    Args:
        qconfig: 量化配置对象
        field_path: 字段路径，用于错误信息
        
    Raises:
        SchemaValidateError: 当校验失败时抛出异常
    """
    is_per_group = qconfig.scope == QScope.PER_GROUP
    has_group_size = "group_size" in qconfig.ext

    if is_per_group:
        # 当scope为per_group时，必须存在group_size且大于0
        if not has_group_size:
            raise SchemaValidateError(
                f"When quantization config scope is per_group, "
                f"ext field must contain group_size, "
                f"but {field_path} does not have group_size field",
                action=f"Please add group_size parameter to {field_path} ext field"
            )

        group_size = qconfig.ext["group_size"]
        if not isinstance(group_size, int) or group_size <= 0:
            raise SchemaValidateError(
                f"When quantization config scope is per_group, "
                f"group_size in ext field must be a positive integer, "
                f"but {field_path} has group_size={group_size}",
                action=f"Please set {field_path} ext.group_size to a positive integer"
            )
    else:
        # 当scope不为per_group时，不应该存在group_size
        if has_group_size:
            raise SchemaValidateError(
                f"When quantization config scope is not per_group, "
                f"ext field should not contain group_size, but {field_path} has group_size field",
                action=f"Please remove group_size parameter from {field_path} ext field"
            )


class AutoroundProcessorConfig(AutoProcessorConfig):
    type: Literal["autoround_quant"] = Field(default="autoround_quant", description="处理器类型标识")
    iters: int = Field(default=10, gt=0, description="迭代次数，必须大于0")
    enable_minmax_tuning: bool = Field(default=True, description="是否启用最小最大值调优")
    enable_round_tuning: bool = Field(default=True, description="是否启用舍入调优")
    strategies: List[QuantStrategyConfig] = Field(default_factory=list, description="量化策略配置列表")

    @model_validator(mode='after')
    def validate_strategies(self) -> 'AutoroundProcessorConfig':
        """校验strategies字段中的量化配置"""
        _validate_quantization_strategies(self.strategies)

        # 调用create_layer_config进行配置与预检
        for i, strategy in enumerate(self.strategies):
            try:
                _create_layer_config(strategy.qconfig)
            except Exception as e:
                # 明确异常配置发生的位置
                raise SchemaValidateError(
                    f"Configuration validation failed for strategies[{i}]: {str(e)}",
                    action=f"Please check the configuration of strategies[{i}].qconfig and fix the error"
                ) from e

        return self


@QABCRegistry.register(dispatch_key=AutoroundProcessorConfig, abc_class=AutoSessionProcessor)
@logger_setter(prefix="msmodelslim.processor.autoround_quant")
class AutoroundQuantProcessor(AutoSessionProcessor):
    def __init__(
            self,
            model: nn.Module,
            config: AutoroundProcessorConfig,
            adapter: Optional[object] = None,
    ) -> None:

        super().__init__(model)
        self.model = model
        self.config = config

        self.iters = config.iters
        self.enable_minmax_tuning = config.enable_minmax_tuning
        self.enable_round_tuning = config.enable_round_tuning
        self.enable_quanted_input = False
        self.enable_trainable_smooth = False

        self.device = torch.device("cpu")
        self.shared_cache_keys = get_shared_keys(self.model)
        self.layer_config = self.build_layer_config_from_strategies()
        self.apply_layer_config()

        self.float_output = None
        self.quantized_output = None
        self.best_params = None
        self.quantized_layer_names = []
        self.unquantized_layer_names = []

    def support_distributed(self) -> bool:
        return False

    def pre_run(self) -> None:
        # 全局的准备工作
        for n, m in self.model.named_parameters():
            m.requires_grad_(False)
        for n, m in self.model.named_modules():
            if isinstance(m, SUPPORTED_LAYER_TYPES):
                m.name = n

    def preprocess(self, request: BatchProcessRequest) -> None:
        self.device = next(request.module.parameters()).device.type

        self._run_forward_if_need(request)
        self.float_output = [output[0] for output in request.outputs]
        get_logger().debug(f"Captured {len(self.float_output)} float outputs")

        with torch.device(device=self.device):
            self.quantized_layer_names, self.unquantized_layer_names = _wrapper_block(
                request.module, self.enable_minmax_tuning, self.enable_round_tuning, self.enable_trainable_smooth,
                config=self.model.config)

    def process(self, request: BatchProcessRequest) -> None:
        get_logger().info(f"Starting quantization training with {self.iters} iterations")

        trainer = BlockQuantTrainer(
            iters=self.iters,
            enable_minmax_tuning=self.enable_minmax_tuning,
            enable_quanted_input=self.enable_quanted_input,
            shared_cache_keys=self.shared_cache_keys,
            gradient_accumulate_steps=8,
        )
        get_logger().debug(
            f"Trainer initialized with minmax_tuning={self.enable_minmax_tuning}, "
            f"quanted_input={self.enable_quanted_input}")

        input_ids, input_others = _convert_request_datas_format(request.datas)
        get_logger().debug(f"Converted input data: {len(input_ids)} input_ids, {len(input_others)} input_others")

        if self.quantized_output is not None:
            input_ids = self.quantized_output
            get_logger().debug("Using quantized output as input for next layer")

        self.best_params = trainer.train_block(
            block=request.module,
            input_ids=input_ids,
            input_others=input_others,
            float_output=self.float_output,
            device=self.device,
        )

    def postprocess(self, request: BatchProcessRequest) -> None:

        if self.quantized_output is not None:  # 将输入替换为上一层的量化后的结果再跑前向
            get_logger().debug("Replacing input with quantized output from previous layer")
            for data, out_q in zip(request.datas, self.quantized_output):
                data[0][0][0] = out_q

        if self.enable_quanted_input:
            get_logger().debug("Running forward pass with quantized input...")
            self._run_forward_if_need(request)
            self.quantized_output = [output[0] for output in request.outputs]
            request.outputs = [(output_f,) + data[1:] for output_f, data in zip(self.float_output, request.outputs)]
            get_logger().debug(f"Generated {len(self.quantized_output)} quantized outputs")

        # 应用最佳参数
        get_logger().debug("Applying best parameters and unwrapping blocks...")
        with torch.no_grad(), torch.device(device=self.device):
            self._unwrapper_block(request.module, self.best_params)
            self.best_params = {}

    def post_run(self) -> None:
        for n, m in self.model.named_modules():
            if hasattr(m, "name"):
                delattr(m, "name")

    def build_layer_config_from_strategies(self) -> Dict[str, Dict[str, Any]]:
        """根据配置文件中的策略构建layer_config字典"""
        layer_config = {}

        # 获取支持的层
        supported_layers = [
            name
            for name, module in self.model.named_modules()
            if isinstance(module, SUPPORTED_LAYER_TYPES)
        ]

        get_logger().info(f"Found {len(supported_layers)} supported layers for quantization")
        get_logger().debug(f"Supported layers: {supported_layers}")

        if not self.config.strategies:
            get_logger().warning("No quantization strategies configured, all layers will use default configuration")
            return layer_config

        get_logger().info(f"Processing {len(self.config.strategies)} quantization strategies")

        # 处理每个策略
        for i, strategy_config in enumerate(self.config.strategies):
            get_logger().debug(
                f"Processing strategy {i}: include={strategy_config.include}, exclude={strategy_config.exclude}")
            include_set = ConfigSet(strategy_config.include)
            exclude_set = ConfigSet(strategy_config.exclude)

            strategy_applied_count = 0
            for layer_name in supported_layers:
                if _should_apply_strategy(layer_name, include_set, exclude_set, layer_config):
                    layer_config[layer_name] = _create_layer_config(strategy_config.qconfig)
                    strategy_applied_count += 1
                    get_logger().debug(f"Applied strategy {i} to layer '{layer_name}'")

            get_logger().info(f"Strategy {i} applied to {strategy_applied_count} layers")
            _warning_unmatched_pattern(f"{i}.include", include_set)
            _warning_unmatched_pattern(f"{i}.exclude", exclude_set)

        get_logger().info(f"Layer configuration built: {len(layer_config)} layers configured")
        return layer_config

    def apply_layer_config(self) -> None:
        """将layer_config中的配置应用到模型的每一层"""
        configured_count = 0
        default_count = 0

        for name, module in self.model.named_modules():
            if isinstance(module, SUPPORTED_LAYER_TYPES):
                config = self.layer_config.get(name)
                if config:
                    _apply_config_to_module(module, config)
                    configured_count += 1
                    get_logger().debug(f"Applied custom config to layer '{name}': {config}")
                else:
                    _apply_default_config(module)
                    default_count += 1
                    get_logger().debug(f"Applied default config to layer '{name}'")

        get_logger().info(f"Layer configuration applied: {configured_count} custom, {default_count} default")

    @torch.no_grad()
    def _unwrapper_block(self, block: nn.Module, best_params: Dict[str, Any]) -> None:
        """Unwraps the WrapperLinear and WrapperTransformerConv1d modules in the given block.

        Args:
            block: The input block containing wrapped modules to be unwrapped.
            best_params: A dictionary of best parameters for the wrapped modules.
        """
        unwrapped_count = 0

        for n, m in block.named_modules():
            if hasattr(m, "orig_layer"):
                best_param = best_params.get(n)
                get_logger().debug(f"Unwrapping layer '{n}' with best_params: {best_param is not None}")

                orig_layer: torch.nn.Linear = m.unwrapper(best_param)

                # ir保存部分
                with torch.device(device=self.device):
                    fake_quantizer = _create_fake_quantizer(orig_layer)
                    get_logger().debug(f"Created fake quantizer for layer '{n}'")

                    # 应用hooks
                    hook_count = 0
                    for hook in orig_layer._forward_pre_hooks.values():
                        if isinstance(hook, qir.HookIR):
                            fake_quantizer = hook.wrapper_module(fake_quantizer)
                            hook_count += 1

                    if hook_count > 0:
                        get_logger().debug(f"Applied {hook_count} hooks to layer '{n}'")

                    block.set_submodule(n, fake_quantizer)
                    unwrapped_count += 1

        get_logger().info(f"Unwrapped {unwrapped_count} layers")
