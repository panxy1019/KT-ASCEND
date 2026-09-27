#!/usr/bin/env python3
"""Build the immutable two-block Round13 B QSA Decode component bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil

from build_round13_b import patch_context_runtime


VERSIONS = {215: "A", 216: "B"}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def manifest(root):
    return {
        str(path.relative_to(root)): sha(path)
        for path in sorted(root.rglob("*"))
        if path.is_file()
        and path.name != "source-sha256.json"
        and "__pycache__" not in path.parts
    }


def build(*, parent, inputs, output, versions=VERSIONS):
    parent, inputs, output = map(Path, (parent, inputs, output))
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    shutil.copytree(
        parent / "source/runtime",
        output / "runtime",
        ignore=shutil.ignore_patterns("__pycache__"),
    )
    runtime = output / "runtime/context_reference_runtime.py"
    baseline_runtime_sha = sha(runtime)
    runtime.write_text(patch_context_runtime(runtime.read_text()))
    shutil.copy2(
        parent / "source/tools/probe_identity.py", output / "probe_identity.py"
    )
    shutil.copy2(
        inputs / "round13_b_qsa_component_probe.py",
        output / "round13_b_qsa_component_probe.py",
    )
    contract = {
        "schema": 1,
        "status": "FROZEN_NOT_RUN",
        "stage": "ROUND13_B_DIRECT_BF16_QSA_DECODE_COMPONENT",
        "candidate": "DIRECT_RESIDENT_BF16_ROW_DECODE",
        "versions": {str(version): block for version, block in versions.items()},
        "layers": [3, 23, 47],
        "contexts": [64, 128, 512],
        "iterations": 5,
        "precision": "BF16_INPUT_FP32_ATTENTION",
        "baseline_runtime_sha256": baseline_runtime_sha,
        "candidate_runtime_sha256": sha(runtime),
        "numerical_gate_relative_l2": 0.001,
        "performance_gate_candidate_over_baseline": 0.90,
        "required": [
            "independent sparse_gqa oracle",
            "causal future token rejection",
            "unqualified geometry staged-row fallback",
            "repeat hash exact",
            "batch drain, cancel drain, stale completion fail-stop",
            "allocator cleanup",
            "two independent blocks each improve complete call >=10%",
        ],
        "automatic_retries": 0,
    }
    (output / "CONTRACT.json").write_text(json.dumps(contract, indent=2) + "\n")
    (output / "source-sha256.json").write_text(
        json.dumps(manifest(output), indent=2, sort_keys=True) + "\n"
    )
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--version-a", type=int, default=215)
    parser.add_argument("--version-b", type=int, default=216)
    args = parser.parse_args()
    if args.version_a == args.version_b:
        raise ValueError("component blocks require distinct versions")
    result = build(
        parent=args.parent,
        inputs=args.inputs,
        output=args.output,
        versions={args.version_a: "A", args.version_b: "B"},
    )
    print(
        "PASS_ROUND13_B_COMPONENT_SNAPSHOT",
        sha(result / "source-sha256.json"),
    )


if __name__ == "__main__":
    main()
