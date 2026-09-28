#!/usr/bin/env python3
"""Strictly salvage successful Round15 P0 measurements after a control-only failure."""

from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path

from round15_p0_control import classify_physical12_after


EXPECTED_WORKLOADS = ("WARMUP", "W1", "W2", "DRAIN")
EXPECTED_COMPLETIONS = (16, 16, 128, 4)
EXPECTED_QSA = ("VECTORIZED", "DIRECT_BF16_ROW")


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def manifest(path, prefix):
    rows = {}
    for line in Path(path).read_text().splitlines():
        digest, name = line.split(None, 1)
        name = name.strip()
        if not name.startswith(prefix):
            raise RuntimeError(f"manifest path outside frozen prefix: {name}")
        rows[name[len(prefix):]] = digest
    return rows


def parse_time(value):
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def validate(root):
    root = Path(root).resolve(strict=True)
    evidence = root / "20260928-round15-p0-a-v335"
    admission = root / "v335"
    local = manifest(
        root / "v335-files.sha256",
        "artifacts/round15/p0-results/v335-control-failure/20260928-round15-p0-a-v335/",
    )
    remote = manifest(
        root / "v335-remote-files.sha256",
        "/mnt/data1/kt-round9/evidence/20260928-round15-p0-a-v335/",
    )
    if local != remote or len(local) != 251:
        raise RuntimeError("v335 local/remote evidence inventory mismatch")
    for relative, digest in local.items():
        if sha256(evidence / relative) != digest:
            raise RuntimeError(f"v335 local evidence changed: {relative}")

    raw = json.loads((evidence / "round15-p0-raw.json").read_text())
    if (
        raw.get("status") != "PASS_ROUND15_P0_PROFILE_DRIVER_RAW"
        or raw.get("stage") != "ROUND15_P0_FRESH_PROMOTED_S0_PROFILE"
        or (raw.get("version"), raw.get("block")) != (335, "A")
        or raw.get("generation_attempts") != 4
        or raw.get("abort_controls") != 0
        or raw.get("automatic_retries") != 0
        or raw.get("graph") is not False
    ):
        raise RuntimeError("v335 raw identity, budget or mode mismatch")
    results = raw.get("results", [])
    if tuple(row.get("workload") for row in results) != EXPECTED_WORKLOADS:
        raise RuntimeError("v335 workload schedule mismatch")
    if tuple(row.get("completion_tokens") for row in results) != EXPECTED_COMPLETIONS:
        raise RuntimeError("v335 completion lengths mismatch")
    for row in results:
        if row.get("finite_logprobs") is not True:
            raise RuntimeError(f"v335 non-finite logprob: {row.get('workload')}")
        qsa = row.get("qsa", {})
        if (qsa.get("prefill_strategy"), qsa.get("decode_strategy")) != EXPECTED_QSA:
            raise RuntimeError(f"v335 QSA path mismatch: {row.get('workload')}")
        if qsa.get("actual_completion_tokens") != row.get("completion_tokens"):
            raise RuntimeError(f"v335 QSA completion mismatch: {row.get('workload')}")
        if row.get("path", {}).get("cpu_submit_count", 0) <= 0:
            raise RuntimeError(f"v335 Hybrid CPU path absent: {row.get('workload')}")
        if row.get("path", {}).get("npu_resident_count", 0) <= 0:
            raise RuntimeError(f"v335 Hybrid NPU path absent: {row.get('workload')}")
    for workload in ("W1", "W2"):
        row = next(item for item in results if item["workload"] == workload)
        if row.get("output_ids", [])[:2] != [3897, 248046]:
            raise RuntimeError(f"v335 {workload} ACK/EOS prefix mismatch")

    census = json.loads((evidence / "round14a-five-bucket-memory.json").read_text())
    expected_buckets = {"Expert", "PLE", "Dense backbone", "KV+GDN state", "Workspace"}
    if (
        census.get("status") != "PASS_ROUND14A_FIVE_BUCKET_STORAGE_CENSUS"
        or set(census.get("buckets", {})) != expected_buckets
        or census.get("unique_storage_count", 0) <= 0
        or census.get("unique_storage_bytes", 0) <= 0
    ):
        raise RuntimeError("v335 five-bucket census contract mismatch")
    if any(value.get("unique_storages", 0) <= 0 for value in census["buckets"].values()):
        raise RuntimeError("v335 five-bucket census contains an empty bucket")

    supervisor = json.loads((evidence / "round15-p0-supervisor.json").read_text())
    state = supervisor.get("owned_exit_state", {})
    if (
        supervisor.get("driver_exit") != 0
        or supervisor.get("owned_container_removed") is not True
        or state.get("OOMKilled") is not False
        or not str(supervisor.get("guard_exit_ack", {}).get("action", "")).startswith("OWNED_EXITED")
    ):
        raise RuntimeError("v335 owned-process cleanup proof mismatch")
    discovery = json.loads((admission / "discovery.json").read_text())
    cgroups = {}
    for window in discovery.get("windows", []):
        cgroups.update({int(pid): value for pid, value in window.get("physical12_process_cgroups", {}).items()})
    physical = classify_physical12_after(
        (evidence / "post-cleanup-proc-mem.txt").read_text(), cgroups.__getitem__
    )
    if physical["status"] != "REGISTERED_TRAINING_ONLY":
        raise RuntimeError("v335 post-cleanup device identity was not the registered training task")

    started = parse_time(state["StartedAt"])
    finished = parse_time(state["FinishedAt"])
    seconds = (finished - started).total_seconds()
    if seconds <= 0:
        raise RuntimeError("v335 non-positive device runtime")
    return {
        "schema": 1,
        "status": "PASS_ROUND15_P0_PROFILE_EVIDENCE_SALVAGED_AFTER_CONTROL_FAILURE",
        "version": 335,
        "block": "A",
        "original_raw_status": raw["status"],
        "original_supervisor_status": supervisor["status"],
        "control_failures": [
            {
                "kind": "PROFILER_METADATA_NAME_MISMATCH",
                "raw": raw.get("profiler"),
                "contract": "BOUNDED_EXISTING_EVENT_OBSERVER",
            },
            {
                "kind": "REGISTERED_TRAINING_PROCESS_MISTAKEN_FOR_UNKNOWN_PROCESS",
                "accepted_identity": physical,
            },
        ],
        "measurements": {
            row["workload"]: {
                "completion_tokens": row["completion_tokens"],
                "ttft_seconds": row["ttft_seconds"],
                "tpot_seconds": row["tpot_seconds"],
                "e2e_seconds": row["e2e_seconds"],
                "finite_logprobs": row["finite_logprobs"],
            }
            for row in results
        },
        "five_bucket_memory": {
            "unique_storage_count": census["unique_storage_count"],
            "unique_storage_bytes": census["unique_storage_bytes"],
            "buckets": census["buckets"],
            "allocator": census.get("allocator"),
        },
        "cleanup": {
            "driver_exit": 0,
            "guard_exit_ack": supervisor["guard_exit_ack"],
            "owned_container_removed": True,
            "oom_killed": False,
            "physical12_after": physical,
        },
        "runtime": {"seconds": seconds, "npu_device_hours": seconds / 3600.0},
        "evidence": {
            "files": len(local),
            "local_remote_sha256_exact": True,
            "evidence_tar_sha256": sha256(root / "v335-evidence.tar"),
            "admission_tar_sha256": sha256(root / "v335-admission.tar"),
            "raw_sha256": sha256(evidence / "round15-p0-raw.json"),
            "supervisor_sha256": sha256(evidence / "round15-p0-supervisor.json"),
            "census_sha256": sha256(evidence / "round14a-five-bucket-memory.json"),
        },
        "qualification_boundary": (
            "Measurements are admitted as P0 block-A data; the original control failure remains "
            "preserved and this receipt is not a replacement process PASS."
        ),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    result = validate(args.root)
    output = args.root / "round15-p0-final-salvaged.json"
    if output.exists():
        raise FileExistsError(output)
    output.write_text(json.dumps(result, indent=2) + "\n")
    receipt = {
        "schema": 1,
        "status": "PASS_V335_ARCHIVE_RECEIPT",
        "source": str(args.root),
        "salvaged_result": output.name,
        "salvaged_result_sha256": sha256(output),
        "evidence_tar_sha256": result["evidence"]["evidence_tar_sha256"],
        "admission_tar_sha256": result["evidence"]["admission_tar_sha256"],
        "remote_local_files": result["evidence"]["files"],
        "remote_local_sha256_exact": True,
    }
    archive = args.root / "ARCHIVE_RECEIPT.json"
    if archive.exists():
        raise FileExistsError(archive)
    archive.write_text(json.dumps(receipt, indent=2) + "\n")
    print(result["status"], result["runtime"]["seconds"], flush=True)


if __name__ == "__main__":
    main()
