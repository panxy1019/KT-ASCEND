"""Fail-closed Round16 four-process request comparison.

The comparator deliberately accepts only the frozen ten-generation block
shape.  It computes service throughput from each continuous measurement
window; TPOT is never inverted or reused as a throughput estimate.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from statistics import median
from typing import Any, Mapping, Sequence


class ComparisonValidationError(ValueError):
    """The evidence is incomplete or not comparable, so no speed verdict exists."""


@dataclass(frozen=True)
class PairResult:
    pair_index: int
    control_process_id: str
    candidate_process_id: str
    primary_latency_ratio: float
    primary_throughput_ratio: float | None
    protection_ratios: tuple[tuple[str, float], ...]


@dataclass(frozen=True)
class ComparisonResult:
    status: str
    promoted: bool
    target: str
    pairs: tuple[PairResult, ...]
    reasons: tuple[str, ...]


_REQUESTED_OUTPUT = {"W1": 16, "W2": 128, "W4": 32}
_ROLE_ORDER = ("CONTROL", "CANDIDATE", "CANDIDATE", "CONTROL")


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ComparisonValidationError(f"{name} must be a mapping")
    return value


def _exact_int(value: Any, name: str, expected: int | None = None) -> int:
    if type(value) is not int:
        raise ComparisonValidationError(f"{name} must be an exact integer")
    if expected is not None and value != expected:
        raise ComparisonValidationError(f"{name} must be exactly {expected}")
    return value


def _nonempty_string(value: Any, name: str) -> str:
    if type(value) is not str or not value:
        raise ComparisonValidationError(f"{name} must be a non-empty string")
    return value


def _sha256(value: Any, name: str) -> str:
    value = _nonempty_string(value, name)
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ComparisonValidationError(f"{name} must be lowercase SHA-256")
    return value


def _finite_positive(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise ComparisonValidationError(f"{name} must be a finite positive number")
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise ComparisonValidationError(
            f"{name} must be a finite positive number"
        ) from error
    if not isfinite(number) or number <= 0:
        raise ComparisonValidationError(f"{name} must be a finite positive number")
    return number


def _metric_pair(value: Any, name: str) -> tuple[float, float]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ComparisonValidationError(f"{name} must contain two measurements")
    frozen = tuple(value)
    if len(frozen) != 2:
        raise ComparisonValidationError(f"{name} must contain two measurements")
    return (
        _finite_positive(frozen[0], f"{name}[0]"),
        _finite_positive(frozen[1], f"{name}[1]"),
    )


@dataclass(frozen=True)
class _Workload:
    ttft_ms: tuple[float, float]
    tpot_ms: tuple[float, float]
    e2e_ms: tuple[float, float]
    window_throughput: float


@dataclass(frozen=True)
class _Block:
    role: str
    process_id: str
    model_identity_sha256: str
    gates_pass: bool
    workloads: Mapping[str, _Workload]


def _parse_workload(value: Any, expected_name: str) -> _Workload:
    row = _mapping(value, f"workload {expected_name}")
    if row.get("workload") != expected_name:
        raise ComparisonValidationError(f"workload identity drift for {expected_name}")
    _exact_int(row.get("warmup_attempts"), f"{expected_name}.warmup_attempts", 1)
    _exact_int(
        row.get("measurement_attempts"),
        f"{expected_name}.measurement_attempts",
        2,
    )
    requested = _exact_int(
        row.get("requested_output_tokens"),
        f"{expected_name}.requested_output_tokens",
        _REQUESTED_OUTPUT[expected_name],
    )
    actual = row.get("actual_output_tokens")
    if not isinstance(actual, Sequence) or isinstance(actual, (str, bytes)):
        raise ComparisonValidationError(
            f"{expected_name}.actual_output_tokens must contain two requests"
        )
    actual = tuple(actual)
    if len(actual) != 2 or any(type(item) is not int or item != requested for item in actual):
        raise ComparisonValidationError(
            f"{expected_name} must produce the frozen output count in both measurements"
        )
    _exact_int(
        row.get("measurement_window_output_tokens"),
        f"{expected_name}.measurement_window_output_tokens",
        sum(actual),
    )
    _exact_int(
        row.get("measurement_window_successes"),
        f"{expected_name}.measurement_window_successes",
        2,
    )
    _exact_int(
        row.get("measurement_window_failures"),
        f"{expected_name}.measurement_window_failures",
        0,
    )
    seconds = _finite_positive(
        row.get("measurement_window_seconds"),
        f"{expected_name}.measurement_window_seconds",
    )
    return _Workload(
        ttft_ms=_metric_pair(row.get("ttft_ms"), f"{expected_name}.ttft_ms"),
        tpot_ms=_metric_pair(row.get("tpot_ms"), f"{expected_name}.tpot_ms"),
        e2e_ms=_metric_pair(row.get("e2e_ms"), f"{expected_name}.e2e_ms"),
        window_throughput=sum(actual) / seconds,
    )


def _parse_block(
    value: Any,
    expected_role: str,
    common_identity: str,
) -> _Block:
    row = _mapping(value, f"{expected_role} block")
    if row.get("role") != expected_role:
        raise ComparisonValidationError("four-process role order must be A/B/B/A")
    process_id = _nonempty_string(row.get("process_id"), "process_id")
    if _sha256(row.get("common_identity_sha256"), "block common identity") != common_identity:
        raise ComparisonValidationError("block common identity drift")
    model_identity = _sha256(row.get("model_identity_sha256"), "model identity")
    if row.get("profiler_enabled") is not False:
        raise ComparisonValidationError("performance blocks require profiler OFF")
    _exact_int(row.get("error_count"), "error_count", 0)
    _exact_int(row.get("generation_attempts"), "generation_attempts", 10)
    workloads = _mapping(row.get("workloads"), "workloads")
    if set(workloads) != set(_REQUESTED_OUTPUT):
        raise ComparisonValidationError("every block must contain exactly W1, W2 and W4")
    parsed = {name: _parse_workload(workloads[name], name) for name in _REQUESTED_OUTPUT}
    gates = (
        "quality_pass",
        "state_pass",
        "cancel_reuse_pass",
        "resource_pass",
        "canary_pass",
    )
    for gate in gates:
        if type(row.get(gate)) is not bool:
            raise ComparisonValidationError(f"{gate} must be an explicit boolean")
    return _Block(
        role=expected_role,
        process_id=process_id,
        model_identity_sha256=model_identity,
        gates_pass=all(row[gate] for gate in gates),
        workloads=parsed,
    )


def _ratio(candidate: Sequence[float], control: Sequence[float]) -> float:
    return median(candidate) / median(control)


def _pair(
    pair_index: int,
    control: _Block,
    candidate: _Block,
    target: str,
) -> PairResult:
    if target == "DECODE":
        primary_latency = _ratio(
            candidate.workloads["W2"].tpot_ms,
            control.workloads["W2"].tpot_ms,
        )
        primary_throughput = (
            candidate.workloads["W2"].window_throughput
            / control.workloads["W2"].window_throughput
        )
        protected = ("W1", "W4")
    else:
        primary_latency = _ratio(
            candidate.workloads["W1"].ttft_ms,
            control.workloads["W1"].ttft_ms,
        )
        primary_throughput = None
        protected = ("W2", "W4")
    protections: list[tuple[str, float]] = []
    for name in protected:
        protections.append(
            (
                f"{name}.TTFT",
                _ratio(
                    candidate.workloads[name].ttft_ms,
                    control.workloads[name].ttft_ms,
                ),
            )
        )
        protections.append(
            (
                f"{name}.E2E",
                _ratio(
                    candidate.workloads[name].e2e_ms,
                    control.workloads[name].e2e_ms,
                ),
            )
        )
    return PairResult(
        pair_index=pair_index,
        control_process_id=control.process_id,
        candidate_process_id=candidate.process_id,
        primary_latency_ratio=primary_latency,
        primary_throughput_ratio=primary_throughput,
        protection_ratios=tuple(protections),
    )


def evaluate_abba(document: Mapping[str, Any]) -> ComparisonResult:
    """Validate and evaluate one frozen Round16 A/B/B/A comparison."""

    document = _mapping(document, "comparison document")
    _exact_int(document.get("schema"), "schema", 1)
    _nonempty_string(document.get("comparison_id"), "comparison_id")
    if document.get("comparison_kind") not in ("A_SAME_RUNTIME", "B_SYSTEM"):
        raise ComparisonValidationError("comparison_kind is not a Round16 contract")
    common_identity = _sha256(
        document.get("common_identity_sha256"), "common_identity_sha256"
    )
    target_contract = _mapping(document.get("target_contract"), "target_contract")
    if target_contract.get("frozen_before_first_process") is not True:
        raise ComparisonValidationError("primary target was not frozen before execution")
    target = target_contract.get("target")
    if target == "DECODE":
        expected = (
            "W2",
            "TPOT",
            "CONTINUOUS_WINDOW_OUTPUT_TOKENS_PER_SECOND",
        )
    elif target == "PREFILL":
        expected = ("W1", "TTFT_AND_E2E", "NONE")
    else:
        raise ComparisonValidationError("target must be DECODE or PREFILL")
    observed = (
        target_contract.get("primary_workload"),
        target_contract.get("primary_latency_metric"),
        target_contract.get("primary_throughput_metric"),
    )
    if observed != expected:
        raise ComparisonValidationError("target metric identity drift")

    raw_blocks = document.get("blocks")
    if not isinstance(raw_blocks, Sequence) or isinstance(raw_blocks, (str, bytes)):
        raise ComparisonValidationError("blocks must be an A/B/B/A sequence")
    if len(raw_blocks) != 4:
        raise ComparisonValidationError("exactly four independent process blocks are required")
    blocks = tuple(
        _parse_block(value, role, common_identity)
        for value, role in zip(raw_blocks, _ROLE_ORDER)
    )
    process_ids = tuple(block.process_id for block in blocks)
    if len(set(process_ids)) != 4:
        raise ComparisonValidationError("all four model processes must be independent")
    if blocks[0].model_identity_sha256 != blocks[3].model_identity_sha256:
        raise ComparisonValidationError("control model identity drift across ABBA")
    if blocks[1].model_identity_sha256 != blocks[2].model_identity_sha256:
        raise ComparisonValidationError("candidate model identity drift across ABBA")

    pairs = (
        _pair(1, blocks[0], blocks[1], target),
        _pair(2, blocks[3], blocks[2], target),
    )
    if not all(block.gates_pass for block in blocks):
        return ComparisonResult(
            "REJECTED_CORRECTNESS_OR_RESOURCE",
            False,
            target,
            pairs,
            ("CORRECTNESS_STATE_CANCEL_OR_RESOURCE_GATE_FAILED",),
        )

    reasons: list[str] = []
    if target == "DECODE":
        if any(pair.primary_latency_ratio > 0.90 for pair in pairs):
            reasons.append("PRIMARY_LATENCY_GATE_FAILED")
        if any(
            pair.primary_throughput_ratio is None
            or pair.primary_throughput_ratio < 1.10
            for pair in pairs
        ):
            reasons.append("PRIMARY_THROUGHPUT_GATE_FAILED")
    else:
        if any(pair.primary_latency_ratio > 0.85 for pair in pairs):
            reasons.append("PRIMARY_LATENCY_GATE_FAILED")
        for control, candidate in ((blocks[0], blocks[1]), (blocks[3], blocks[2])):
            if _ratio(
                candidate.workloads["W1"].e2e_ms,
                control.workloads["W1"].e2e_ms,
            ) > 0.90:
                reasons.append("PRIMARY_E2E_GATE_FAILED")
                break
    if any(ratio > 1.05 for pair in pairs for _name, ratio in pair.protection_ratios):
        reasons.append("PROTECTION_GATE_FAILED")
    reasons = list(dict.fromkeys(reasons))
    if reasons:
        return ComparisonResult(
            "MEASURED_NO_PROMOTION", False, target, pairs, tuple(reasons)
        )
    return ComparisonResult("QUALIFIED_REQUEST_GAIN", True, target, pairs, ())


__all__ = (
    "ComparisonResult",
    "ComparisonValidationError",
    "PairResult",
    "evaluate_abba",
)
