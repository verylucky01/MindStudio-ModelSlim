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
"""
from typing import Any, Dict, List, Literal, Optional, Tuple

import torch
from pydantic import Field
from torch import nn

from msmodelslim.core.base.protocol import BatchProcessRequest
from msmodelslim.core.context import get_current_context
from msmodelslim.ir.qal.qregistry import QABCRegistry
from msmodelslim.processor.base import AutoProcessorConfig, AutoProcessorConfigList, AutoSessionProcessor
from msmodelslim.utils.logging import get_logger
from msmodelslim.utils.exception import UnexpectedError, UnsupportedError

from .metrics.factory import ModelWiseMethodFactory

def _require_hidden_tensor(
    obj: Any,
) -> torch.Tensor:
    """Extract the first hidden tensor from common layer-wise I/O shapes."""
    t: Optional[torch.Tensor] = None

    if isinstance(obj, torch.Tensor):
        t = obj
    elif isinstance(obj, tuple) and obj:
        head, *rest = obj
        if isinstance(head, torch.Tensor):
            t = head
        # (args, kwargs) row
        elif rest and isinstance(rest[0], dict) and isinstance(head, tuple) and head and isinstance(head[0], torch.Tensor):
            t = head[0]

    if t is None:
        raise UnexpectedError("Failed to extract hidden_states tensor.")
    return t

class BinaryOperatorModelWiseProcessorConfig(AutoProcessorConfig):
    """模型级敏感层分析配置（对比模型最终输出，支持多种 metrics 如 MSE）"""

    type: Literal["binary_operator_model_wise"] = "binary_operator_model_wise"
    metrics: str = Field(
        default="mse_model_wise",
        description="分析方法：mse",
    )
    patterns: List[str] = Field(
        default_factory=lambda: ["*"],
        description=(
            "与 linear_quant.include 建议一致（如 YAML 同用 ${patterns}）；"
            "用于层敏感结果展示名后缀，如 model.layers.2 (*mlp*)。实际量化范围以 linear_quant 为准。"
        ),
    )
    configs: AutoProcessorConfigList = Field(
        default_factory=list,
        description="量化子处理器配置列表，用于进行量化-反量化",
    )


@QABCRegistry.register(dispatch_key=BinaryOperatorModelWiseProcessorConfig, abc_class=AutoSessionProcessor)
class BinaryOperatorModelWiseProcessor(AutoSessionProcessor):
    """模型级敏感层分析"""

    def __init__(
        self,
        model: nn.Module,
        config: BinaryOperatorModelWiseProcessorConfig,
        adapter: Optional[object] = None,
    ):
        super().__init__(model)
        self.config = config
        self.adapter = adapter
        self.quant_processors = [
            AutoSessionProcessor.from_config(model, cfg, adapter)
            for cfg in config.configs
        ]
        self._analysis_method = ModelWiseMethodFactory.create_method(
            config.metrics, adapter=adapter
        )
        self._base_data_count: int = 0
        self._block_names: List[str] = []
        self._float_outputs: List[Any] = []
        self._quant_inputs: List[Any] = []
        self._merged_outputs: List[Any] = []

    def pre_run(self) -> None:
        ctx = get_current_context()
        if ctx is None:
            raise UnexpectedError("No context is working.")
        for processor in self.quant_processors:
            processor.pre_run()

    def preprocess(self, request: BatchProcessRequest) -> None:
        self._block_names.append(request.name)
        request.datas = self._replace_request_datas_with_merged_outputs_if_need(request.datas)

        if self._base_data_count == 0:
            self._base_data_count = len(request.datas)

        float_inputs, quant_inputs = self._build_float_quant_inputs(request.datas)

        request.datas = float_inputs
        self._run_forward_if_need(request)
        self._float_outputs = request.outputs
        self._quant_inputs = quant_inputs

    def process(self, request: BatchProcessRequest) -> None:
        request.datas = self._quant_inputs
        for qp in self.quant_processors:
            qp.preprocess(request)
            qp.process(request)
            qp.postprocess(request)

    def postprocess(self, request: BatchProcessRequest) -> None:
        request.datas = self._quant_inputs
        self._run_forward_if_need(request)

        # 将纯浮点输出与带量化输出结果拼接
        self._merged_outputs = [*self._float_outputs, *request.outputs]

    def post_run(self) -> None:
        for processor in self.quant_processors:
            processor.post_run()

        self._validate_merged_outputs()
        layer_scores = self._compute_layer_scores()

        self._annotate_layer_scores_with_patterns(layer_scores)

        self._write_layer_analysis_debug(layer_scores)

        get_logger().info(
            "BinaryOperatorModelWiseProcessor post_run: %d layer scores (%s), patterns=%s",
            len(layer_scores),
            self._analysis_method.name,
            self.config.patterns,
        )

    def _validate_merged_outputs(self) -> None:
        base_count = self._base_data_count
        num_layers = len(self._block_names)
        expected = base_count * (num_layers + 1) if base_count > 0 else 0

        if base_count <= 0 or len(self._merged_outputs) < expected:
            raise UnexpectedError(
                "BinaryOperatorModelWiseProcessor post_run got invalid merged outputs: "
                f"base_count={base_count}, merged={len(self._merged_outputs)}, "
                f"num_layers={num_layers}, expected={expected}."
            )

    def _compute_layer_scores(self) -> List[Dict[str, Any]]:
        layer_scores: List[Dict[str, Any]] = []
        base_count = self._base_data_count
        ref_outputs: List[Any] = self._merged_outputs[:base_count]
        for layer_idx, layer_name in enumerate(self._block_names):
            block_base = base_count * (layer_idx + 1)
            cand_outputs: List[Any] = self._merged_outputs[block_base:block_base + base_count]

            score = self._analysis_method.compute_score(ref_outputs, cand_outputs)
            layer_scores.append(
                {
                    "name": layer_name,
                    "score": score,
                }
            )
        return layer_scores

    def _write_layer_analysis_debug(self, layer_scores: List[Dict[str, Any]]) -> None:
        ctx = get_current_context()
        if ctx is None:
            return
        ctx["layer_analysis"].debug["layer_scores"] = layer_scores
        ctx["layer_analysis"].debug["method"] = self._analysis_method.name
        ctx["layer_analysis"].debug["patterns"] = list(self.config.patterns)

    def _replace_request_datas_with_merged_outputs_if_need(
        self, datas: Optional[List[Tuple[tuple, dict]]]
    ) -> Optional[List[Tuple[tuple, dict]]]:
        """若存在上一层 merged_outputs，则用其 hidden_states 重建当前层 datas。"""
        merged_outputs = self._merged_outputs
        base_data_count = self._base_data_count

        if not merged_outputs or datas is None:
            return datas

        old_rows = datas

        if base_data_count > 0:
            if len(old_rows) < base_data_count or len(merged_outputs) < base_data_count:
                raise UnexpectedError(
                    "BinaryOperatorModelWiseProcessor got inconsistent tensor counts for hidden_states "
                    f"consistency check: base_data_count={base_data_count}, "
                    f"datas={len(old_rows)}, merged_outputs={len(merged_outputs)}."
                )

            base_rows = old_rows[:base_data_count]
            tail_outputs = merged_outputs[-base_data_count:]
            for idx, (row, out) in enumerate(zip(base_rows, tail_outputs)):
                req_hidden = _require_hidden_tensor(row)
                merged_hidden = _require_hidden_tensor(out)

                merged_hidden = merged_hidden.to(
                    device=req_hidden.device,
                    dtype=req_hidden.dtype,
                )
                if req_hidden.shape != merged_hidden.shape or not torch.allclose(req_hidden, merged_hidden):
                    raise UnsupportedError(
                        "Current model does not support model-wise sensitive layer analysis: "
                        "during forward, current layer input hidden_states != previous layer output."
                    )

        new_rows: List[Tuple[tuple, dict]] = []
        for idx, out in enumerate(merged_outputs):
            hidden = _require_hidden_tensor(out)

            # 与 layer_wise_forward 的约定一致：args[0] 为 hidden_states。
            _, template_kwargs = old_rows[idx % len(old_rows)]
            new_rows.append(((hidden,), template_kwargs))

        return new_rows

    def _build_float_quant_inputs(
        self,
        datas: Optional[List[Tuple[tuple, dict]]],
    ) -> Tuple[List[Tuple[tuple, dict]], List[Tuple[tuple, dict]]]:
        """float 用全部行，quant 用前 ``quant_source_count`` 行。"""
        num_datas = self._base_data_count or None
        return list(datas), datas[:num_datas]

    def _annotate_layer_scores_with_patterns(self, layer_scores: List[Dict[str, Any]]) -> None:
        """写入展示用 ``name``（如 ``model.layers.2 (*mlp*)``），原始 block 路径写入 ``module_name``。"""
        patterns = self.config.patterns
        pat = ", ".join(patterns) if patterns else "*"
        for row in layer_scores:
            base = row.get("name", "")
            row["module_name"] = base
            row["name"] = f"{base} ({pat})"