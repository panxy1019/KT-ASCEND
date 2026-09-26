#!/usr/bin/env python3
"""Build the immutable Round11 v144 query-aware vectorized-QSA snapshot."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil


VERSION = 144


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build(output: Path, *, version=VERSION):
    if version != VERSION:
        raise ValueError(f"QSA component snapshot must use version {VERSION}")
    if output.exists():
        raise FileExistsError(output)
    repo = Path(__file__).resolve().parents[2]
    output.mkdir(parents=True)
    (output / "runtime").mkdir()
    copies = {
        repo / "migration_bridge/round9/runtime/qsa.py": output / "runtime/qsa.py",
        repo / "migration_bridge/round9/tools/probe_identity.py": output / "probe_identity.py",
        repo / "artifacts/round11/round11_t5_qsa_component_probe.py": output / "round11_t5_qsa_component_probe.py",
        repo / "artifacts/round11/round11_t5_qsa_component_controller.py": output / "round11_t5_qsa_component_controller.py",
        repo / "artifacts/round11/t2-results/v115/source/round11_monitor_owned.py": output / "round11_monitor_owned.py",
        repo / "artifacts/round11/t2-results/v115/source/numa_policy.py": output / "numa_policy.py",
        repo / "artifacts/round11/t2-admissions/v115-attempt1/round10_admission.py": output / "round11_admission.py",
    }
    for source, destination in copies.items():
        shutil.copy2(source, destination)
    contract = {
        "schema": 1,
        "status": "FROZEN_NOT_RUN",
        "stage": "ROUND11_T5_QSA_SINGLE_CANDIDATE_COMPONENT",
        "version": VERSION,
        "parent_identity": "ROUND11_HFINAL_V107_ORIGINAL_C128_MXFP4",
        "hotspot_ranking_sha256": _sha(repo / "artifacts/round11/ROUND11_T5_HOTSPOT_RANKING.json"),
        "candidate": "QUERY_AWARE_VECTORIZED_SPARSE_GQA",
        "changed_runtime_files": ["runtime/qsa.py"],
        "unchanged": [
            "QSA indexer", "KV pool", "selected token IDs", "precision",
            "provider count", "Hybrid threshold", "resident map", "PLE", "Dynamic",
            "Decode sparse GQA row implementation",
        ],
        "fixtures": {
            "layers": [3, 23, 47],
            "prefill_context_tokens": [32, 64, 128],
            "query_tokens": 32,
            "decode_context_tokens": 129,
            "decode_query_tokens": 1,
            "weights": "REAL_CHECKPOINT",
            "hidden": "DETERMINISTIC_SYNTHETIC",
        },
        "gates": {
            "relative_l2_max": 0.001,
            "repeat_hashes": 3,
            "each_prefill_context_ratio_max": 0.90,
            "decode_ratio_max": 1.05,
        },
        "iterations_per_side": 9,
        "automatic_retries": 0,
        "repair_lineage": {
            "v141": "PRELAUNCH_GUARD_MISSING_NUMA_POLICY_ZERO_COMPONENT_EXECUTION",
            "v142": "COMPONENT_COMPLETE_PROBE_CLEANUP_FAIL_UNUSED_GATE_REFERENCE",
            "v143": "CLEAN_COMPONENT_QUERY_AGNOSTIC_DECODE_RATIO_1_059338_NO_PROMOTION",
            "v144_changes": [
                "STAGE_EXISTING_GUARD_NUMA_POLICY_DEPENDENCY",
                "DO_NOT_RETAIN_UNUSED_GATE_TENSOR_VIEW_IN_PROBE",
                "USE_VECTORIZED_PATH_ONLY_WHEN_QUERY_TOKENS_EXCEED_ONE",
            ],
        },
    }
    (output / "ROUND11_T5_QSA_COMPONENT_CONTRACT.json").write_text(
        json.dumps(contract, indent=2) + "\n"
    )
    manifest = {
        str(path.relative_to(output)): _sha(path)
        for path in sorted(output.rglob("*")) if path.is_file()
    }
    (output / "source-sha256.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--version", type=int, default=VERSION)
    args = parser.parse_args()
    built = build(args.output, version=args.version)
    print(built, _sha(built / "source-sha256.json"), flush=True)


if __name__ == "__main__":
    main()
