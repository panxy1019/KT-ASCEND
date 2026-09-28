#!/usr/bin/env python3
"""Frozen Round14a incremental-residency profiles.

Both profiles add exactly 480 BF16 experts to Round14's Uniform-Q32 parent.
They change startup placement only; no request-time migration is introduced.
"""

from __future__ import annotations

import hashlib
import json


LAYERS = 48
EXPERTS_PER_LAYER = 512
BASELINE_PER_LAYER = 32
ADDED_EXPERTS = 480
BUDGET_EXPERTS = LAYERS * BASELINE_PER_LAYER + ADDED_EXPERTS
EXPERT_BF16_BYTES = 3 * 2560 * 640 * 2
NOMINAL_BF16_BYTES = BUDGET_EXPERTS * EXPERT_BF16_BYTES


def _digest(rows: list[list[int]]) -> str:
    encoded = json.dumps(rows, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def validate_rows(rows: list[list[int]]) -> list[list[int]]:
    if not isinstance(rows, list) or len(rows) != LAYERS:
        raise ValueError("Round14a resident placement requires exactly 48 rows")
    normalized = []
    for layer, row in enumerate(rows):
        if not isinstance(row, list) or row != sorted(row) or len(row) != len(set(row)):
            raise ValueError(f"Round14a resident row {layer} is not sorted and unique")
        if any(type(expert) is not int or not 0 <= expert < EXPERTS_PER_LAYER for expert in row):
            raise ValueError(f"Round14a resident row {layer} contains an invalid expert")
        normalized.append(list(row))
    if sum(map(len, normalized)) != BUDGET_EXPERTS:
        raise ValueError("Round14a resident placement is not the frozen 2016-expert budget")
    return normalized


def fplus1() -> list[list[int]]:
    rows = [list(range(EXPERTS_PER_LAYER))]
    rows.extend(list(range(BASELINE_PER_LAYER)) for _ in range(LAYERS - 1))
    return validate_rows(rows)


def uplus42() -> list[list[int]]:
    # 2016 / 48 = 42 exactly: ten additional experts in every layer.
    return validate_rows([list(range(42)) for _ in range(LAYERS)])


def _receipt(name: str, rows: list[list[int]]) -> dict:
    rows = validate_rows(rows)
    counts = list(map(len, rows))
    return {
        "name": name,
        "parent": "ROUND14_S0_UNIFORM_Q32",
        "budget_experts": BUDGET_EXPERTS,
        "added_experts_vs_parent": ADDED_EXPERTS,
        "nominal_bf16_bytes": NOMINAL_BF16_BYTES,
        "nominal_bf16_gib": NOMINAL_BF16_BYTES / 2**30,
        "per_layer_counts": counts,
        "selected_logical_experts": rows,
        "profile_sha256": _digest(rows),
    }


def build_document() -> dict:
    return {
        "schema": 1,
        "status": "FROZEN_ROUND14A_INCREMENTAL_RESIDENCY_PROFILES",
        "selection_basis": (
            "FIVE_BUCKET_CAPACITY_GATE_K_SAFE_1_WITH_1GIB_GDN_AND_C4_STATE_RESERVES"
        ),
        "same_nominal_budget": True,
        "profiles": {
            "fplus1": _receipt("fplus1", fplus1()),
            "uplus42": _receipt("uplus42", uplus42()),
        },
    }


def main() -> None:
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.write_text(json.dumps(build_document(), indent=2) + "\n")
    print("PASS_ROUND14A_RESIDENT_PROFILES", BUDGET_EXPERTS, NOMINAL_BF16_BYTES)


if __name__ == "__main__":
    main()

