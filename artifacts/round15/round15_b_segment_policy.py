"""Fail-closed eligibility policy for Round15's exact resident-MoE NPU segment."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence


@dataclass(frozen=True)
class SegmentDecision:
    use_graph: bool
    reason: str
    key: Optional[tuple] = None


def _shape(value: Sequence[int] | None) -> tuple[int, ...] | None:
    if value is None:
        return None
    return tuple(int(item) for item in value)


def decide_resident_moe_graph(
    *,
    enabled: bool,
    forward_mode: object,
    hidden_shape: Sequence[int] | None,
    topk_ids_shape: Sequence[int] | None,
    topk_weights_shape: Sequence[int] | None,
    hidden_dtype: str,
    ids_dtype: str,
    weights_dtype: str,
    layer_idx: int,
    tp_size: int,
    num_gpu_experts: int,
    method_name: str,
    return_fp32_contribution: bool,
    with_bias: bool,
    max_running_requests: int,
) -> SegmentDecision:
    """Admit only the frozen serial Decode geometry used by Round15.

    A one-token EXTEND can have the same tensor shape as Decode, so shape alone
    is deliberately insufficient.  Unknown metadata always falls back to eager.
    """

    if not enabled:
        return SegmentDecision(False, "DISABLED")
    mode = getattr(forward_mode, "name", forward_mode)
    mode = str(mode).upper() if mode is not None else "UNKNOWN"
    if mode != "DECODE":
        return SegmentDecision(False, f"FORWARD_MODE_{mode}")
    if _shape(hidden_shape) != (1, 2560):
        return SegmentDecision(False, "HIDDEN_SHAPE")
    if _shape(topk_ids_shape) != (1, 10):
        return SegmentDecision(False, "TOPK_IDS_SHAPE")
    if _shape(topk_weights_shape) != (1, 10):
        return SegmentDecision(False, "TOPK_WEIGHTS_SHAPE")
    if hidden_dtype != "torch.bfloat16":
        return SegmentDecision(False, "HIDDEN_DTYPE")
    # The frozen Qwen3.8 SGLang Decode path emits int32 routed-expert IDs.
    # Keep this exact so an unverified route dtype falls back to eager.
    if ids_dtype != "torch.int32":
        return SegmentDecision(False, "IDS_DTYPE")
    if weights_dtype != "torch.float32":
        return SegmentDecision(False, "WEIGHTS_DTYPE")
    if type(layer_idx) is not int or not 0 <= layer_idx < 48:
        return SegmentDecision(False, "LAYER_INDEX")
    if tp_size != 1:
        return SegmentDecision(False, "TP_SIZE")
    if num_gpu_experts != 32:
        return SegmentDecision(False, "RESIDENT_CAPACITY")
    if method_name != "UnquantizedFusedMoEMethod":
        return SegmentDecision(False, "PROVIDER")
    if not return_fp32_contribution:
        return SegmentDecision(False, "HYBRID_ACCUMULATION_DTYPE")
    if with_bias:
        return SegmentDecision(False, "BIAS")
    if max_running_requests != 1:
        return SegmentDecision(False, "SERIAL_OWNERSHIP")
    return SegmentDecision(
        True,
        "QUALIFIED_DECODE_BS1_Q32_BF16_NZ29_TOPK_IDS_INT32",
        (
            layer_idx,
            (1, 2560),
            (1, 10),
            hidden_dtype,
            ids_dtype,
            weights_dtype,
            num_gpu_experts,
        ),
    )


__all__ = ("SegmentDecision", "decide_resident_moe_graph")
