"""
-------------------------------------------------------------------------
This file is part of the MindStudio project.
Copyright (c) 2026 Huawei Technologies Co.,Ltd.

MindStudio is licensed under Mulan PSL v2.
You may use this software according to the terms and conditions of the Mulan PSL v2.
You may obtain a copy of Mulan PSL v2 at:

         http://license.coscl.org.cn/MulanPSL2

THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
See the Mulan PSL v2 for more details.
-------------------------------------------------------------------------
"""
from abc import abstractmethod
from typing import Any, Dict, List, Optional

from msmodelslim.processor.analysis.methods_base import LayerAnalysisMethod


class ModelWiseAnalysisMethod(LayerAnalysisMethod):
    """模型级敏感层分析方法基类：仅指标语义（``compute_score``）。

    **Processor**：双路径 batch 构造、路径合并、前向编排；block 输出对齐为张量由 Processor 内建逻辑完成。

    **各 metrics 模块**（如 ``mse.py``）：实现 ``compute_score``。

    扩展新指标：在 ``metrics`` 下新增 ``<name>.py`` 并在 ``factory`` 注册；若编排相同可复用当前 Processor。
    """

    @abstractmethod
    def compute_score(
        self,
        final_outputs: List[Any],
        block_names: List[str],
        base_data_count: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """根据最后一层合并列表计算各层敏感分数。

        典型布局：``len(final_outputs) == base_count * (len(block_names) + 1)``，
        前 ``base_count`` 条为各样本纯浮点参考，之后每层连续 ``base_count`` 条为该层各样本输出。
        ``base_data_count`` 由 Processor 传入以便校验，可为 ``None`` 则仅按长度推断。
        """

    def get_hook(self) -> Any:
        """兼容 LayerAnalysisMethod；本分析类型不使用子模块 forward hook。"""
        return None
