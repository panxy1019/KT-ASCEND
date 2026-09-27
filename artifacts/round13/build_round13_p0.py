#!/usr/bin/env python3
"""Build immutable local Round13 P0 contracts from qualified Round12a evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import time

from round13_placement import (
    BUDGET_EXPERTS,
    NOMINAL_BF16_BYTES,
    RANDOM_ALGORITHM,
    build_profiles,
    profile_receipt,
    random_global_budget,
)


MODEL_REVISION = "de4b8e4d43b917e7706784d8bb445c9af86a3540"
QSA_SHA256 = "0fd8d98b37f7e90e90f9866356892236dcc7d0114351d1438025561df6c1bd35"
QSA_CONSUMER_SHA256 = "d4583c5abe9791b09c48d1dd8f8491a2988eb5147d45df188e0e48908c699f1f"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def _run_identity(root: Path, version: int) -> dict:
    evidence = root / "artifacts" / "round12a" / "remote-evidence" / f"v{version}-abba"
    final_path = evidence / "round12a-model-final.json"
    final = json.loads(final_path.read_text())
    if (
        final.get("status") != "PASS_ROUND12A_MODEL_PROCESS"
        or final.get("mode") != "KT"
        or final.get("version") != version
    ):
        raise ValueError(f"Round12a v{version} is not a qualified KT parent")
    return {
        "qualified": True,
        "evidence": str(evidence.relative_to(root)),
        "final_sha256": sha256(final_path),
        "framework_profile_sha256": sha256(evidence / "framework-profile.json"),
        "server_profile_sha256": sha256(evidence / "server-profile.json"),
        "q32_physical_sha256": sha256(evidence / "q32-physical.json"),
        "payload_sha256_receipt_sha256": sha256(evidence / "payload-sha256.json"),
        "source_manifest_sha256": sha256(evidence / "SHA256SUMS.txt"),
        "remote_snapshot": f"/mnt/data1/kt-round9/round12a-kt-{'a' if version == 174 else 'b'}-v{version}",
    }


def _copy_inputs(root: Path, out: Path) -> dict[str, dict]:
    sources = {
        "round10-t1-fixture.json": root / "artifacts" / "round10" / "round10-t1-fixture.json",
        "round11-t1-fixture.json": root / "artifacts" / "round11" / "round11-t1-fixture.json",
        "qwen38_sched_dev.json": root / "artifacts" / "round12a" / "remote-evidence" / "v174-abba" / "source" / "runtime" / "profiles" / "qwen38_sched_dev.json",
        "qwen38_sched_holdout.json": root / "artifacts" / "round12a" / "remote-evidence" / "v174-abba" / "source" / "runtime" / "profiles" / "qwen38_sched_holdout.json",
        "frequency-q32-source.json": root / "migration_bridge" / "round9" / "evidence" / "physical15" / "20260921-framework-frequency-v1" / "frequency-q32.json",
    }
    frozen = out / "frozen-inputs"
    frozen.mkdir()
    receipt = {}
    for name, source in sources.items():
        if not source.is_file():
            raise FileNotFoundError(source)
        target = frozen / name
        shutil.copyfile(source, target)
        receipt[name] = {
            "source": str(source.relative_to(root)),
            "sha256": sha256(target),
            "bytes": target.stat().st_size,
        }
    return receipt


def _workloads(out: Path, inputs: dict[str, dict]) -> dict:
    round10 = json.loads((out / "frozen-inputs" / "round10-t1-fixture.json").read_text())
    round11 = json.loads((out / "frozen-inputs" / "round11-t1-fixture.json").read_text())
    sources = {
        "edge64": round10["cases"]["edge64"],
        "edge128": round11["cases"]["edge128"],
        "edge512": round11["cases"]["edge512"],
    }
    expected_lengths = {"edge64": 64, "edge128": 128, "edge512": 512}
    for name, expected in expected_lengths.items():
        if len(sources[name]["input_ids"]) != expected:
            raise ValueError(f"{name} fixture length changed")
    common = {
        "temperature": 0,
        "top_p": 1,
        "sampling_seed": 20260920,
        "ignore_eos": True,
        "stream": True,
        "return_logprob": True,
        "logprob_start_len": -1,
    }
    definitions = {
        "W0": ("round10-t1-fixture.json", "edge64", 64, 2, 129, "ROUND12A_W1_BRIDGE"),
        "W1": ("round11-t1-fixture.json", "edge128", 128, 16, 129, "PREFILL_TTFT"),
        "W2": ("round11-t1-fixture.json", "edge128", 128, 128, 129, "DECODE_TPOT"),
        "W3": ("round11-t1-fixture.json", "edge512", 512, 16, 513, "LONGER_PREFILL_EXPERIMENTAL"),
        "W4": ("round10-t1-fixture.json", "edge64", 64, 32, 129, "SHORT_CONTROL"),
    }
    workloads = {}
    for name, (fixture, case, input_tokens, output_tokens, threshold, purpose) in definitions.items():
        workloads[name] = {
            "fixture": fixture,
            "fixture_sha256": inputs[fixture]["sha256"],
            "case_id": case,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "hybrid_threshold": threshold,
            "path": "HYBRID",
            "purpose": purpose,
            "sampling": dict(common, max_new_tokens=output_tokens),
        }
    return {
        "schema": 1,
        "status": "ROUND13_WORKLOADS_FROZEN",
        "model_revision": MODEL_REVISION,
        "workloads": workloads,
        "dev": {
            "fixture": "qwen38_sched_dev.json",
            "fixture_sha256": inputs["qwen38_sched_dev.json"]["sha256"],
            "split": "QWEN38_SCHED_DEV",
            "use": "PLACEMENT_COUNTS_ONLY",
            "prefill_decode_counts_separate": True,
        },
        "holdout": {
            "fixture": "qwen38_sched_holdout.json",
            "fixture_sha256": inputs["qwen38_sched_holdout.json"]["sha256"],
            "split": "QWEN38_SCHED_HOLDOUT",
            "use": "QUALITY_ONLY_NEVER_MASK_TUNING",
        },
        "quality_cases": [
            "canary_marker",
            "canary_math391",
            "canary_json",
            "canary_quote",
            "canary_transform",
        ],
        "metrics": {
            "ttft": {
                "definition": "first real output token monotonic time minus request submission monotonic time",
            },
            "tpot": {
                "definition": "(last token time - first token time)/(actual output tokens - 1)",
                "unknown_when_fewer_than_two_tokens": True,
                "network_batch_is_token_clock": False,
            },
            "serial_output_tokens_per_second": {
                "definition": "qualified actual output tokens divided by continuous measurement-window wall time",
                "scope": "MEASUREMENT_WINDOW",
                "includes_request_handoffs_failures_and_timeouts": True,
                "excludes_warmup_quality_startup": True,
            },
            "requests_per_second": {
                "definition": "successful requests divided by the same measurement-window wall time",
            },
        },
        "quality_and_performance_separation": "performance uses ignore_eos=true to exercise the frozen decode budget; quality requests retain normal EOS",
    }


def _budget() -> dict:
    limits = {
        "P0": {"model_processes": 2, "generation_attempts": 32, "abort_controls": 0, "component_processes": 0},
        "A": {"model_processes": 8, "generation_attempts": 192, "abort_controls": 4, "component_processes": 8},
        "B": {"model_processes": 6, "generation_attempts": 96, "abort_controls": 2, "component_processes": 6},
        "C": {"model_processes": 6, "generation_attempts": 160, "abort_controls": 2, "component_processes": 4},
        "F": {"model_processes": 4, "generation_attempts": 96, "abort_controls": 2, "component_processes": 0},
        "repair_reserve": {"model_processes": 2, "generation_attempts": 24, "abort_controls": 1, "component_processes": 2},
        "total": {"model_processes": 28, "generation_attempts": 600, "abort_controls": 11, "component_processes": 20},
    }
    used = {stage: {key: 0 for key in row} for stage, row in limits.items()}
    return {
        "schema": 1,
        "status": "ROUND13_BUDGET_OPEN",
        "limits": limits,
        "used": used,
        "automatic_retries": 0,
        "unchanged_retry_limit": 0,
        "modified_validations_per_root_cause": 2,
        "device_hours_limit": 36,
        "active_execution_hours_limit": 60,
        "wall_time_seconds": {
            "component": 1800,
            "model": 21600,
            "request": 1800,
            "admission": 900,
            "model_cleanup_reserve": 600,
            "component_cleanup_reserve": 180,
        },
    }


def build(root: Path, out: Path) -> None:
    root = Path(root).resolve()
    out = Path(out)
    sentinels = (
        "BASELINE_MANIFEST.json",
        "WORKLOADS.json",
        "PLACEMENT_PROFILES.json",
        "ROUND13_EXECUTION_BUDGET.json",
        "ROUND13_RESOURCE_CONTRACT.json",
        "P0_INPUT_RECEIPT.json",
        "ROUND13_CURRENT_EXECUTION.md",
        "frozen-inputs",
    )
    if out.exists() and any((out / name).exists() for name in sentinels):
        raise FileExistsError(out)
    out.mkdir(parents=True, exist_ok=True)
    inputs = _copy_inputs(root, out)
    parent_manifest_path = root / "artifacts" / "round12a" / "ROUND12A_BASELINE_MANIFEST.json"
    parent_status_path = root / "artifacts" / "round12a" / "ROUND12A_FINAL_STATUS.json"
    parent_resource_path = root / "artifacts" / "round12a" / "ROUND12A_RESOURCE_CONTRACT.json"
    parent_manifest = json.loads(parent_manifest_path.read_text())
    parent_status = json.loads(parent_status_path.read_text())
    if (
        parent_manifest["model_revision"] != MODEL_REVISION
        or parent_status["status"] != "ROUND12A_COMPLETE_KT_STYLE_VALUE_CONFIRMED"
        or parent_status["decision"] != "KT_STYLE_HETEROGENEOUS_EXECUTION_WORTH_IT_IN_QUALIFIED_SCOPE"
    ):
        raise ValueError("Round12a parent identity is not the qualified KT baseline")
    v174 = _run_identity(root, 174)
    v177 = _run_identity(root, 177)
    if v174["framework_profile_sha256"] != v177["framework_profile_sha256"] or v174["q32_physical_sha256"] != v177["q32_physical_sha256"]:
        raise ValueError("Round12a KT parents do not share one compute identity")

    branch = subprocess.check_output(["git", "branch", "--show-current"], cwd=root, text=True).strip()
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    baseline = {
        "schema": 1,
        "status": "ROUND13_P0_BASELINE_IDENTITY_FROZEN",
        "created_at": time.time(),
        "model": {"revision": MODEL_REVISION, "path": "/models/" + MODEL_REVISION},
        "repository": {"branch": branch, "head": head, "dirty_runtime_snapshot_required": True},
        "image": parent_manifest["image"],
        "configuration": {
            "cpu": "KUNPENG_C128_4X32",
            "numa_nodes": [2, 3, 4, 7],
            "cpu_bank": "READ_ONLY_MXFP4",
            "npu_resident": "STATIC_BF16_NZ29",
            "resident_budget_experts": BUDGET_EXPERTS,
            "resident_nominal_bf16_bytes": NOMINAL_BF16_BYTES,
            "scheduler_chunk": 32,
            "qsa": "PROMOTED_QUERY_AWARE_VECTORIZED_SPARSE_GQA",
            "qsa_sha256": QSA_SHA256,
            "qsa_consumer_sha256": QSA_CONSUMER_SHA256,
            "ple": "ASYNC",
            "hybrid_threshold_default": 129,
            "dynamic": "OFF",
            "deferred": 0,
            "r2": "OFF",
            "r3": "OFF",
            "transpose_view": "OFF",
            "shadow320": "OFF",
            "tp": 1,
            "concurrency": 1,
        },
        "parent_runs": {"v174": v174, "v177": v177},
        "parent_artifacts": {
            "baseline_manifest": {"path": str(parent_manifest_path.relative_to(root)), "sha256": sha256(parent_manifest_path)},
            "final_status": {"path": str(parent_status_path.relative_to(root)), "sha256": sha256(parent_status_path)},
            "resource_contract": {"path": str(parent_resource_path.relative_to(root)), "sha256": sha256(parent_resource_path)},
        },
        "frozen_inputs": inputs,
    }
    write_json(out / "BASELINE_MANIFEST.json", baseline)
    write_json(out / "WORKLOADS.json", _workloads(out, inputs))

    source_frequency = json.loads((out / "frozen-inputs" / "frequency-q32-source.json").read_text())
    if (
        source_frequency.get("identity", {}).get("model_manifest_sha256")
        != "62e6eca0f2900745dbfd9ba6ed76e60547e0d602acf2d62a9ecac3972028b38e"
        or source_frequency.get("statistics", {}).get("tokens") != 47
    ):
        raise ValueError("compatible historical DEV route statistics changed")
    counts = source_frequency["statistics"]["counts"]
    profiles = build_profiles(counts)
    placement = {
        "schema": 1,
        "status": "ROUND13_PLACEMENT_BOOTSTRAP_COMPATIBLE_DEV_PENDING_FRESH_P0",
        "budget": {
            "experts": BUDGET_EXPERTS,
            "nominal_bf16_bytes": NOMINAL_BF16_BYTES,
            "nominal_bf16_gib": NOMINAL_BF16_BYTES / 2**30,
            "actual_nz_padding_metadata_loading_peak": "PENDING_COMPONENT_AND_MODEL_RECEIPTS",
        },
        "frequency": {
            "definition": "GLOBAL_TOP1536_OVER_LAYER_EXPERT",
            "tie_break": ["count_desc", "layer_asc", "expert_id_asc"],
            "bootstrap_source": inputs["frequency-q32-source.json"],
            "bootstrap_tokens": source_frequency["statistics"]["tokens"],
            "fresh_p0_prefill_decode_counts": "NOT_RUN",
        },
        "front_loading": {"definition": "LAYERS_0_1_2_Q512_REMAINING_Q0"},
        "random": {
            "definition": "GLOBAL_WITHOUT_REPLACEMENT",
            "algorithm": RANDOM_ALGORITHM,
            "primary_seed": 42,
            "offline_sensitivity_seeds": [43, 44],
            "selection_policy": "SEED42_ONLY_NO_BEST_SEED_SELECTION",
        },
        "profiles": {name: profile_receipt(name, profile, counts) for name, profile in profiles.items()},
        "random_sensitivity": {
            str(seed): profile_receipt(f"random_seed{seed}", random_global_budget(seed), counts)
            for seed in (43, 44)
        },
    }
    write_json(out / "PLACEMENT_PROFILES.json", placement)
    write_json(out / "ROUND13_EXECUTION_BUDGET.json", _budget())

    resource = json.loads(parent_resource_path.read_text())
    resource["schema"] = 2
    resource["round"] = 13
    resource["parent_round12a_sha256"] = sha256(parent_resource_path)
    resource["admission_failure_cycles_limit"] = 6
    resource["cache_retirement"] = {
        "hook": "artifacts/round11/kt_cache_retirement_hook.py",
        "after_owned_exit_guard_ack_archive_full_sha_only": True,
        "registered_read_only_files_only": True,
        "whole_host_fd_and_mmap_audit": True,
        "global_drop_caches": False,
    }
    write_json(out / "ROUND13_RESOURCE_CONTRACT.json", resource)
    write_json(out / "ROUND13_EXECUTION_BUDGET.json", _budget())
    write_json(out / "P0_INPUT_RECEIPT.json", {"schema": 1, "status": "PASS_ROUND13_P0_INPUT_FREEZE", "files": inputs})
    (out / "ROUND13_CURRENT_EXECUTION.md").write_text(
        "# Round13 Current Execution\n\n"
        "- Status: `P0_IDENTITY_FROZEN_PROFILE_RUNS_NOT_STARTED`\n"
        "- Parent: Round12a v174/v177 qualified KT compute identity.\n"
        "- Next: build two independent fresh S0 profiling snapshots for W2/W3 and DEV route counts.\n"
        "- Device processes started: 0.\n"
        "- Generation attempts used: 0.\n"
        "- Automatic retries: 0.\n"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--out", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    build(args.root, args.out)
    print("PASS_ROUND13_P0_BUILD", sha256(args.out / "BASELINE_MANIFEST.json"), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
