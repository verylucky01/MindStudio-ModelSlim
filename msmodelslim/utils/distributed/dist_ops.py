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

from enum import Enum
from typing import Union, List

import torch
import torch.distributed as dist


from msmodelslim.utils.logging import get_logger
from msmodelslim.utils.exception import UnsupportedError


class ReduceOperation(str, Enum):
    """
    Supported reduce operations for distributed synchronization.
    
    Attributes:
        MIN: Find minimum value across all processes
        MAX: Find maximum value across all processes
        SUM: Sum values across all processes
        MEAN: Calculate mean value across all processes
        PROD: Calculate product across all processes
    """
    MIN = "min"
    MAX = "max"
    SUM = "sum"
    MEAN = "mean"
    PROD = "prod"


def sync_base_operation(tensor: torch.Tensor, op: Union[ReduceOperation, str], group=None) -> torch.Tensor:
    """
    执行不增加显存开销的分布式原地操作
    
    支持的操作为: min、max、sum、mean、prod
    
    Args:
        tensor: 要操作的张量，操作结果会原地更新到此张量中
        op: 操作类型，可以使用 ReduceOperation 枚举或字符串
            - ReduceOperation.MIN / 'min': 最小值
            - ReduceOperation.MAX / 'max': 最大值
            - ReduceOperation.SUM / 'sum': 求和
            - ReduceOperation.MEAN / 'mean': 平均值
            - ReduceOperation.PROD / 'prod': 乘积
        group: 进程组，默认为 None（使用默认进程组）
    
    Returns:
        原地更新后的张量（与输入tensor是同一个对象）
    
    """
    
    # Convert string to enum if necessary
    if isinstance(op, str):
        try:
            op = ReduceOperation(op.lower())
        except ValueError as e:
            raise UnsupportedError(
                f"Unsupported operation: {op}. "
                f"Supported operations are: {', '.join([operation.value for operation in ReduceOperation])}",
            ) from e
    
    # Perform the operation based on enum value
    if op == ReduceOperation.MIN:
        dist.all_reduce(tensor, op=dist.ReduceOp.MIN, group=group)
    elif op == ReduceOperation.MAX:
        dist.all_reduce(tensor, op=dist.ReduceOp.MAX, group=group)
    elif op == ReduceOperation.SUM:
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM, group=group)
    elif op == ReduceOperation.PROD:
        dist.all_reduce(tensor, op=dist.ReduceOp.PRODUCT, group=group)
    elif op == ReduceOperation.MEAN:
        # 当前hccl后端不支持，通过 mean = sum / world_size 计算结果
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM, group=group)
        world_size = dist.get_world_size(group)
        tensor.div_(world_size)
    
    return tensor


def sync_gather_tensors(
    tensor: torch.Tensor, 
    variable_shapes: bool = False,
    on_cpu: bool = False,
    group: dist.ProcessGroup = None
) -> list:
    """
    在所有进程间收集张量，如果在npu上进行会增加显存开销
    
    Args:
        tensor: 要收集的本地张量
        variable_shapes: 是否支持不同形状的张量聚合（仅在 on_cpu=False 时有效）
                        - False: 所有进程的张量形状必须相同（更快，默认）
                        - True: 支持不同形状（需要额外通信开销）
        on_cpu: 聚合操作在哪里进行
                - False: 在 NPU 上进行聚合（使用 HCCL，默认）
                - True: 在 CPU 上进行聚合（避免 NPU 显存溢出）
        group: 进程组，默认为 None（使用默认进程组）
    
    Returns:
        收集到的张量列表，列表长度为 world_size
        - 如果 on_cpu=False: 张量在 NPU 上
        - 如果 on_cpu=True: 张量在 CPU 上
    """
    
    world_size = dist.get_world_size(group)
    rank = dist.get_rank(group)
    
    if on_cpu:
        # 场景：在 CPU 上进行聚合（针对大张量避免 NPU 显存溢出）        
        tensor_cpu = tensor.cpu()
        tensor_list = [None] * world_size
        dist.all_gather_object(tensor_list, tensor_cpu, group=group)
        get_logger().debug(f"Gathered {world_size} tensors on CPU")
        return tensor_list
    
    else:
        # 场景：在 NPU 上进行聚合（使用 HCCL 高效通信）
        if variable_shapes:
            with torch.device(tensor.device):
                # 同步张量形状
                local_shape = torch.tensor(tensor.shape, dtype=torch.long)
                shape_list = [torch.zeros_like(local_shape) for _ in range(dist.get_world_size())]
                dist.all_gather(shape_list, local_shape)

                # 初始化存储
                tensor_list = [
                    torch.zeros(*s.tolist(), dtype=tensor.dtype)
                    for s in shape_list
                ]

                # 收集数据
                dist.all_gather(tensor_list, tensor)
                get_logger().debug(f"Gathered {world_size} tensors with variable shapes on NPU using HCCL")
                return tensor_list
        
        else:
            # NPU 上聚合 - 相同形状（最快路径）
            tensor_list = [torch.zeros_like(tensor) for _ in range(world_size)]
            dist.all_gather(tensor_list, tensor, group=group)
            
            get_logger().debug(f"Gathered {world_size} tensors with same shape on NPU using HCCL")
        
        return tensor_list


def sync_gather_tensor_lists(
    tensor_list: List[torch.Tensor],
    group: dist.ProcessGroup = None
) -> List[torch.Tensor]:
    """
    在所有进程间收集张量列表，并展平成一个列表
    
    用于收集每个进程上的 tensor 列表（如校准集的 tensor 列表），
    然后将所有进程的列表合并展平成一个统一的列表。
    
    例如：4 卡，每张卡有 12 个 tensor，收集后返回 48 个 tensor 的列表。
    
    Args:
        tensor_list: 本地进程的张量列表
        group: 进程组，默认为 None（使用默认进程组）
    
    Returns:
        所有进程的张量展平后的列表
    """    
    world_size = dist.get_world_size(group)
    gathered_tensor_lists = [None] * world_size
    dist.all_gather_object(gathered_tensor_lists, tensor_list, group=group)
    
    # 展平所有进程的 tensor 列表
    flattened_tensors = []
    for tensor_list in gathered_tensor_lists:
        if tensor_list:
            for t in tensor_list:
                if t is not None:
                    flattened_tensors.append(t)
    
    get_logger().debug(
        "Gathered and flattened %d tensors from %d ranks",
        len(flattened_tensors), world_size
    )
    
    return flattened_tensors

