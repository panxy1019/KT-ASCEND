"""Device-independent contract for the Round18 native resident partial probe."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


LOGICAL_EXPERTS = 512
PHYSICAL_NPU_SLOTS = 32
TOP_K = 10


@dataclass(frozen=True)
class NativePartialCase:
    name: str
    rows: tuple[tuple[int, ...], ...]
    tie_topk: bool = False

    def __post_init__(self) -> None:
        rows = tuple(tuple(row) for row in self.rows)
        if not rows:
            raise ValueError("native partial case must contain at least one token")
        for row in rows:
            if len(row) != TOP_K or len(set(row)) != TOP_K:
                raise ValueError("each Top-K route must contain ten distinct experts")
            if any(type(value) is not int or not 0 <= value < LOGICAL_EXPERTS for value in row):
                raise ValueError("route expert is outside the logical domain")
        object.__setattr__(self, "rows", rows)


def frozen_native_partial_cases() -> tuple[NativePartialCase, ...]:
    """Return the frozen P1 matrix without inferring routes from test output."""

    return (
        NativePartialCase("all_resident", (tuple(range(10)),)),
        NativePartialCase("all_nonresident", (tuple(range(32, 42)),)),
        NativePartialCase("mixed_1_9", ((0, 32, 33, 34, 35, 36, 37, 38, 39, 40),)),
        NativePartialCase("mixed_5_5", ((0, 32, 1, 33, 2, 34, 3, 35, 4, 36),)),
        NativePartialCase("mixed_9_1", ((0, 1, 2, 3, 4, 5, 6, 7, 8, 32),)),
        NativePartialCase(
            "multi_token",
            (
                tuple(range(10)),
                (0, 32, 1, 33, 2, 34, 3, 35, 4, 36),
                tuple(range(32, 42)),
            ),
        ),
        NativePartialCase("topk_ties", (tuple(range(10, 20)),), tie_topk=True),
    )


def validate_native_partial_receipt(receipt: Mapping[str, Any]) -> None:
    required = {
        "status": "PASS_NATIVE_RESIDENT_PARTIAL",
        "device_executed": True,
        "native_runner_preserved": True,
        "native_quant_method": "AscendUnquantizedFusedMoEMethod",
        "logical_experts": LOGICAL_EXPERTS,
        "physical_npu_slots": PHYSICAL_NPU_SLOTS,
        "selector_calls_per_execution": 1,
        "route_weight_application": "EXACTLY_ONCE",
        "shared_finalize_residual_in_partial": False,
    }
    for key, expected in required.items():
        if receipt.get(key) != expected:
            label = "selector" if key == "selector_calls_per_execution" else key
            raise ValueError(f"native partial receipt {label} contract failed")
    cases = receipt.get("cases")
    if not isinstance(cases, list):
        raise ValueError("native partial receipt has no case results")
    expected_names = {case.name for case in frozen_native_partial_cases()}
    observed_names = {case.get("case") for case in cases if isinstance(case, Mapping)}
    if observed_names != expected_names:
        raise ValueError("native partial receipt case matrix is incomplete")
    if any(case.get("passed") is not True or case.get("repeat_exact") is not True for case in cases):
        raise ValueError("native partial receipt contains a failed or unstable case")


__all__ = (
    "LOGICAL_EXPERTS",
    "NativePartialCase",
    "PHYSICAL_NPU_SLOTS",
    "TOP_K",
    "frozen_native_partial_cases",
    "validate_native_partial_receipt",
)
