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
from typing import List, Optional, Literal, Dict, Any

from pydantic import Field
from torch import nn

from msmodelslim.core.QAL.qregistry import QABCRegistry
from msmodelslim.core.base.protocol import BatchProcessRequest
from msmodelslim.quant.processor.base import AutoSessionProcessor, AutoProcessorConfig
from msmodelslim.utils.config_map import ConfigSet
from msmodelslim.utils.logging import get_logger, logger_setter
from msmodelslim.quant.processor.sparse.admm import AdmmPruner
from msmodelslim.quant.ir.w16a16s import W16A16sLinear


class FloatSparseProcessorConfig(AutoProcessorConfig):
    """浮点稀疏处理器配置类：继承自自动处理器配置基类"""
    type: Literal["float_sparse"] = "float_sparse"
    sparse_ratio: float = Field(default=0.3, ge=0.0, le=1.0, description="Sparse ratio")
    include: List[str] = Field(default_factory=list, description="Included module names")
    exclude: List[str] = Field(default_factory=list, description="Excluded module names")


def _warning_unmatched_pattern(name: str, config_set: ConfigSet) -> None:
    """
    警告未匹配的模式
    
    Args:
        name: 模式名称
        config_set: 配置集合
    """
    unmatched_keys = config_set.unmatched_keys()
    unmatched_keys = list(filter(lambda x: x != "*", unmatched_keys))
    if unmatched_keys:
        get_logger().warning(
            f"These {name} patterns are not matched any module, please ensure this is as expected: {unmatched_keys}")


@QABCRegistry.register(dispatch_key=FloatSparseProcessorConfig, abc_class=AutoSessionProcessor)
@logger_setter("msmodelslim.quant.processor.sparse.float_sparse")
class FloatSparseProcessor(AutoSessionProcessor):
    """
    浮点稀疏处理器：实现基于ADMM算法的模型稀疏化
    
    该处理器通过以下步骤实现模型稀疏化：
    1. 在预处理阶段安装hook收集激活统计信息
    2. 在后处理阶段应用ADMM稀疏算法
    3. 对稀疏化后的权重进行量化，保持重要位置的精度
    4. 将处理后的模块部署为量化模块
    """
    
    def __init__(
            self,
            model: nn.Module,
            config: FloatSparseProcessorConfig,
            adapter: Optional[object] = None,
    ):
        """
        初始化浮点稀疏处理器
        
        Args:
            model: 待处理的模型
            config: 处理器配置
            adapter: 适配器对象
        """
        super().__init__(model)
        self.config = config
        self.include = ConfigSet(config.include) if config.include else ConfigSet(["*"])
        self.exclude = ConfigSet(config.exclude) if config.exclude else ConfigSet([])
        
        # ADMM稀疏器字典，key为模块名称，value为AdmmPruner实例
        self.admm_pruners: Dict[str, AdmmPruner] = {}
        # 存储hook句柄，用于后续清理
        self.hook_handles: Dict[str, Any] = {}

    def is_data_free(self) -> bool:
        """返回False表示需要校准集"""
        return False

    def support_distributed(self) -> bool:
        return False

    def post_run(self) -> None:
        """运行后处理：检查未匹配的模式并发出警告"""
        _warning_unmatched_pattern("include", self.include)
        _warning_unmatched_pattern("exclude", self.exclude)

    def preprocess(self, request: BatchProcessRequest) -> None:
        """
        预处理阶段：安装前向hook，收集激活统计信息
        
        Args:
            request: 批处理请求
        """
        get_logger().info(
            f"Float sparse preprocessing module: {request.name}, "
            f"float sparse ratio: {self.config.sparse_ratio}"
        )
        self._install_extract_input_hook(request.name, request.module)

    def postprocess(self, request: BatchProcessRequest) -> None:
        """
        后处理阶段：应用ADMM稀疏算法，然后更新输出
        
        Args:
            request: 批处理请求
        """
        get_logger().info(
            f"Float sparse postprocessing module: {request.name}, "
            f"float sparse ratio: {self.config.sparse_ratio}"
        )

        # 卸载输入提取hook
        self._uninstall_extract_input_hook()

        # 应用稀疏化
        self._apply_sparse(request.name, request.module)
        
        # 再次运行前向传播，更新request.outputs作为下一层的输入
        self._run_forward_if_need(request)
        
        # 释放ADMM稀疏器内存
        self._free_admm_pruners()

        # 部署处理后的模块
        self._deploy(request.name, request.module)

    def _install_extract_input_hook(self, prefix: str, module: nn.Module) -> None:
        """
        为所有符合条件的Linear模块安装前向hook
        
        Args:
            prefix: 模块前缀名称
            module: 待处理的模块
        """
        for name, submodule in module.named_modules():
            full_name = f"{prefix}.{name}" if prefix != "" else name

            # 只处理Linear模块
            if not isinstance(submodule, nn.Linear):
                continue

            # 检查是否在包含列表中
            if full_name not in self.include:
                continue

            # 检查是否在排除列表中
            if full_name in self.exclude:
                continue

            # 处理符合条件的Linear模块
            self._process_linear(full_name, submodule)

    def _process_linear(self, full_name: str, module: nn.Linear) -> None:
        """
        为单个Linear模块创建ADMM稀疏器并安装hook
        
        Args:
            full_name: 模块的完整名称
            module: Linear模块
        """
        # 创建ADMM稀疏器
        admm_pruner = AdmmPruner(module)
        self.admm_pruners[full_name] = admm_pruner
        
        # 安装前向hook
        def forward_hook(nn_module, inputs, outputs):
            """
            前向hook：收集输入输出数据并更新ADMM稀疏器
            
            Args:
                module: 模块对象
                inputs: 输入数据
                outputs: 输出数据
            """
            # 收集输入数据
            if inputs and len(inputs) > 0:
                inp = inputs[0]
            
            # 更新ADMM稀疏器
            if full_name in self.admm_pruners:
                admm_pruner = self.admm_pruners[full_name]
                if inp is not None:
                    admm_pruner.add_batch(inp.detach())
        
        # 注册hook并保存句柄
        hook_handle = module.register_forward_hook(forward_hook)
        self.hook_handles[full_name] = hook_handle

        get_logger().debug(f"Installed forward hook for module {full_name}")

    def _apply_sparse(self, prefix: str, module: nn.Module) -> None:
        """
        应用ADMM稀疏算法到指定模块
        
        Args:
            prefix: 模块前缀名称
            module: 待处理的模块
        """
        
        for name, _ in module.named_modules():
            full_name = f"{prefix}.{name}" if prefix != "" else name
            
            # 检查是否有对应的ADMM稀疏器
            if full_name not in self.admm_pruners:
                continue
                
            admm_pruner = self.admm_pruners[full_name]

            # Execute ADMM sparsification
            get_logger().debug(f"Executing ADMM sparsification: {full_name}")
            admm_pruner.fasterprune(
                sparse_ratio=self.config.sparse_ratio,
            )

            get_logger().debug(f"Completed ADMM sparsification: {full_name}")

    def _deploy(self, prefix: str, module: nn.Module) -> None:
        """
        部署处理后的模块：将稀疏化后的模块转换为量化模块
        
        Args:
            prefix: 模块前缀名称
            module: 待部署的模块
        """
        
        for name, submodule in module.named_modules():
            full_name = f"{prefix}.{name}" if prefix != "" else name
            if not isinstance(submodule, nn.Linear):
                continue

            if full_name not in self.include:
                continue

            if full_name in self.exclude:
                continue

            # 将稀疏化后的Linear模块转换为W16A16s量化模块
            self.model.set_submodule(full_name, W16A16sLinear(submodule.weight, submodule.bias))
            get_logger().debug(f"Replaced module {full_name} with W16A16s module")

    def _free_admm_pruners(self) -> None:
        """释放所有ADMM稀疏器的内存"""
        for full_name in list(self.admm_pruners.keys()):
            admm_pruner = self.admm_pruners[full_name]
            if admm_pruner is not None:
                admm_pruner.free()
        
        self.admm_pruners.clear()

    def _uninstall_extract_input_hook(self) -> None:
        """卸载所有前向hook"""
        for full_name in list(self.hook_handles.keys()):
            hook_handle = self.hook_handles[full_name]
            if hook_handle is not None:
                hook_handle.remove()
        
        self.hook_handles.clear()
