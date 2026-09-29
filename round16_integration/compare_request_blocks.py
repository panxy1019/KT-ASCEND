"""Strict paired comparison for the four Round16 S0/B request blocks."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Mapping


WORKLOADS = ("W1", "W2", "W4")
EXPECTED_BLOCKS = ("S0-A", "B-A", "B-B", "S0-B")


def _validate_block(payload: Mapping[str, Any], expected: str) -> None:
    expected_status = (
        "PASS_ROUND16_S0_REQUEST_PERFORMANCE_BLOCK"
        if expected.startswith("S0")
        else "PASS_ROUND16_B_REQUEST_PERFORMANCE_BLOCK"
    )
    if payload.get("status") != expected_status or payload.get("block") != expected:
        raise ValueError(f"{expected} status or block identity is invalid")
    if payload.get("profiler") != "OFF":
        raise ValueError(f"{expected} was not measured with profiler OFF")
    if payload.get("generation_attempts") != 10:
        raise ValueError(f"{expected} generation count changed")
    if payload.get("automatic_retries") != 0:
        raise ValueError(f"{expected} used an automatic retry")
    summaries = payload.get("summaries")
    if not isinstance(summaries, Mapping) or set(summaries) != set(WORKLOADS):
        raise ValueError(f"{expected} workload summaries are incomplete")
    measurements = [
        row
        for row in payload.get("results", ())
        if row.get("kind") == "performance" and row.get("phase") == "measurement"
    ]
    if len(measurements) != 6:
        raise ValueError(f"{expected} measurement request count changed")
    for row in measurements:
        if row.get("completion_tokens") != row.get("decode_tokens"):
            raise ValueError(f"{expected} output budget changed")
        token_ids = row.get("token_ids")
        if not isinstance(token_ids, list) or token_ids[:2] != [3897, 248046]:
            raise ValueError(f"{expected} ACK/EOS prefix changed")
        selected = row.get("selected_logprobs")
        if selected is not None and (
            len(selected) != row["completion_tokens"]
            or any(not math.isfinite(float(value)) for value in selected)
        ):
            raise ValueError(f"{expected} has invalid selected logprobs")
        finite = row.get("finite_logprobs")
        if finite is not None and finite is not True:
            raise ValueError(f"{expected} has non-finite logprobs")
    for workload in WORKLOADS:
        summary = summaries[workload]
        expected_tokens = {"W1": 32, "W2": 256, "W4": 64}[workload]
        if summary.get("measurement_requests") != 2:
            raise ValueError(f"{expected} {workload} pair count changed")
        if summary.get("output_tokens") != expected_tokens:
            raise ValueError(f"{expected} {workload} output token count changed")
        for metric in (
            "median_tpot_seconds",
            "serial_output_tokens_per_second",
            "median_e2e_seconds",
            "median_ttft_seconds",
        ):
            if not math.isfinite(float(summary[metric])) or float(summary[metric]) <= 0:
                raise ValueError(f"{expected} {workload} {metric} is invalid")


def _pair(control: Mapping[str, Any], candidate: Mapping[str, Any]) -> dict[str, Any]:
    compared: dict[str, Any] = {}
    for workload in WORKLOADS:
        base = control["summaries"][workload]
        new = candidate["summaries"][workload]
        compared[workload] = {
            "s0": dict(base),
            "b16": dict(new),
            "tpot_ratio": new["median_tpot_seconds"] / base["median_tpot_seconds"],
            "throughput_ratio": (
                new["serial_output_tokens_per_second"]
                / base["serial_output_tokens_per_second"]
            ),
            "e2e_ratio": new["median_e2e_seconds"] / base["median_e2e_seconds"],
            "ttft_ratio": new["median_ttft_seconds"] / base["median_ttft_seconds"],
        }
    return compared


def compare_blocks(
    s0_a: Mapping[str, Any],
    b_a: Mapping[str, Any],
    b_b: Mapping[str, Any],
    s0_b: Mapping[str, Any],
) -> dict[str, Any]:
    payloads = (s0_a, b_a, b_b, s0_b)
    for payload, expected in zip(payloads, EXPECTED_BLOCKS, strict=True):
        _validate_block(payload, expected)
    if b_a.get("fixture_sha256") != b_b.get("fixture_sha256"):
        raise ValueError("B block fixture identities differ")
    if b_a.get("model_revision") != b_b.get("model_revision"):
        raise ValueError("B block model revisions differ")
    pairs = (_pair(s0_a, b_a), _pair(s0_b, b_b))
    decode_gate = all(
        pair["W2"]["tpot_ratio"] <= 0.90
        and pair["W2"]["throughput_ratio"] >= 1.10
        for pair in pairs
    )
    protection_gate = all(
        pair[workload][metric] <= 1.05
        for pair in pairs
        for workload in ("W1", "W4")
        for metric in ("ttft_ratio", "e2e_ratio")
    )
    promoted = decode_gate and protection_gate
    return {
        "schema": 1,
        "status": "PASS_ROUND16_STRICT_REQUEST_COMPARISON",
        "comparison_kind": "CROSS_FRAMEWORK_SYSTEM_COMPARISON",
        "pair_order": ["S0-A/B-A", "S0-B/B-B"],
        "pairs": list(pairs),
        "decode_gate": {
            "w2_tpot_ratio_max": 0.90,
            "w2_throughput_ratio_min": 1.10,
        },
        "protection_gate": {"w1_w4_ttft_e2e_ratio_max": 1.05},
        "decode_gate_pass": decode_gate,
        "protection_gate_pass": protection_gate,
        "promoted": promoted,
        "disposition": (
            "B16_REQUEST_GAIN_QUALIFIED" if promoted else "B16_MEASURED_NO_PROMOTION"
        ),
        "fixture_sha256": b_a.get("fixture_sha256"),
        "model_revision": b_a.get("model_revision"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--s0-a", type=Path, required=True)
    parser.add_argument("--b-a", type=Path, required=True)
    parser.add_argument("--b-b", type=Path, required=True)
    parser.add_argument("--s0-b", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = compare_blocks(
        json.loads(args.s0_a.read_text()),
        json.loads(args.b_a.read_text()),
        json.loads(args.b_b.read_text()),
        json.loads(args.s0_b.read_text()),
    )
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(result["disposition"], json.dumps(result["pairs"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
