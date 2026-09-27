#!/usr/bin/env python3
"""Strict four-process Round12a KT/non-KT ABBA comparator."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import statistics


ROLES = {
    174: ("KT", "A"),
    175: ("NON_KT", "A"),
    176: ("NON_KT", "B"),
    177: ("KT", "B"),
}
QSA_LAYERS = list(range(3, 48, 4))
W1_EXPECTED_OUTPUT_IDS = [3897, 248046]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text().splitlines()
        if line.strip()
    ]


def _cgroup_peak_bytes(rows: list[dict]) -> int:
    values = []
    for row in rows:
        cgroup = row.get("owned_cgroup") or {}
        raw = cgroup.get("memory.max_usage_in_bytes")
        if raw is not None:
            values.append(int(str(raw).strip()))
    return max(values, default=0)


def load_run(path: Path, version: int) -> dict:
    path = Path(path)
    return {
        "path": str(path),
        "path_manifest": {
            str(item.relative_to(path)): sha256(item)
            for item in sorted(path.rglob("*"))
            if item.is_file()
        },
        "final": json.loads((path / "round12a-model-final.json").read_text()),
        "raw": json.loads((path / "round12a-model-raw.json").read_text()),
        "supervisor": json.loads(
            (path / "round12a-model-supervisor.json").read_text()
        ),
        "monitor": _read_jsonl(path / "monitor.jsonl"),
        "version": version,
    }


def _median(rows: list[dict], field: str) -> float:
    return statistics.median(row[field] for row in rows)


def validate_run(version: int, item: dict, queue: dict) -> dict:
    mode, block = ROLES[version]
    final = item["final"]
    raw = item["raw"]
    expected_count = 4 * len(queue["workloads"])
    if (
        final.get("status") != "PASS_ROUND12A_MODEL_PROCESS"
        or raw.get("status") != "PASS_ROUND12A_MODEL_DRIVER_RAW"
        or item["supervisor"].get("status") != "PASS_ROUND12A_MODEL_PROCESS"
        or final.get("version") != version
        or final.get("mode") != mode
        or final.get("block") != block
        or final.get("generation_attempts") != expected_count
        or len(final.get("results", [])) != expected_count
        or final.get("automatic_retries") != 0
        or final.get("profiler") != "OFF"
        or final.get("timing_primary") != "CLIENT_UNPROFILED_E2E"
        or final.get("queue_sha256") != queue["sha256"]
        or final.get("cleanup")
        != {
            "guard_exit_ack": True,
            "container_removed": True,
            "physical12_after": "EMPTY",
        }
    ):
        raise ValueError(f"v{version} process/budget/cleanup contract failed")
    binding = final.get("qsa_source_binding", {})
    if binding.get("mode") != "APPROVED_CANDIDATE":
        raise ValueError(f"v{version} did not bind promoted QSA")
    for row in final["results"]:
        qsa = row.get("qsa_dispatch", {})
        if (
            row.get("quality_pass") is not True
            or row.get("finite_logprobs") is not True
            or row.get("finish_reason") != "length"
            or row.get("completion_tokens") != row.get("decode_tokens")
            or row.get("timing_primary") != "CLIENT_UNPROFILED_E2E"
            or row.get("e2e_seconds", float("inf"))
            >= queue.get("request_wall_limit_seconds")
            or qsa.get("mode") != "APPROVED_CANDIDATE"
            or qsa.get("vectorized_layers") != QSA_LAYERS
            or qsa.get("row_layers") != QSA_LAYERS
            or row.get("ple", {}).get("async_batches", 0) <= 0
        ):
            raise ValueError(f"v{version} request quality/provider contract failed")
        if row["workload"] == "W1" and row["output_ids"] != W1_EXPECTED_OUTPUT_IDS:
            raise ValueError(f"v{version} W1 output changed")
        path = row.get("path", {})
        if mode == "KT":
            if (
                path.get("selection_count", 0) <= 0
                or path.get("cpu_submit_count", 0) <= 0
                or path.get("cpu_result_count", 0) <= 0
                or path.get("npu_resident_count", 0) <= 0
            ):
                raise ValueError(f"v{version} KT Hybrid path is incomplete")
        elif path != {
            "selected_path": "NON_KT_OFFLOADERV1",
            "cpu_offload_gib": 233,
            "full_bf16_layers": True,
        }:
            raise ValueError(f"v{version} non-KT path identity changed")
    if mode == "NON_KT":
        receipt = final.get("nonkt_binding", {})
        if (
            receipt.get("status") != "PASS_ROUND12A_NONKT_OFFLOADER_BINDING"
            or receipt.get("decoder_layers") != 48
            or receipt.get("offloader_sha256")
            != "8b68d4012a9bd0fab3bab70ab28db1d7c09b6242af912066d2aedd3f3a3c9006"
            or receipt.get("cpu_parameter_bytes")
            != receipt.get("offloader_accounted_bytes")
        ):
            raise ValueError(f"v{version} non-KT residency identity failed")
    monitor = item["monitor"]
    if not monitor:
        raise ValueError(f"v{version} resource monitor is empty")
    peak_hbm = max(row.get("hbm_used_mib", 0) for row in monitor)
    peak_owned = max(row.get("owned_process_mib", 0) for row in monitor)
    if peak_hbm > 56832 or peak_owned > 37376:
        raise ValueError(f"v{version} NPU resource ceiling failed")
    return {
        "version": version,
        "mode": mode,
        "block": block,
        "evidence_manifest_sha256": hashlib.sha256(
            json.dumps(item["path_manifest"], sort_keys=True).encode()
        ).hexdigest(),
        "startup_seconds": final["startup"]["seconds"],
        "peak_hbm_mib": peak_hbm,
        "peak_owned_npu_mib": peak_owned,
        "peak_host_cgroup_gib": _cgroup_peak_bytes(monitor) / 1024**3,
        "min_host_available_gib": min(
            row.get("host_mem_available_kib", 0) for row in monitor
        )
        / 1024**2,
    }


def compare(items: dict[int, dict], queue_path: Path) -> dict:
    raw_queue = Path(queue_path).read_bytes()
    queue = json.loads(raw_queue)
    if (
        queue.get("schema") != 1
        or queue.get("status") != "FROZEN_ROUND12A_ABBA_QUEUE"
        or queue.get("source_smoke_version") != 173
        or queue.get("workloads") != ["W1"]
        or queue.get("workload_definitions")
        != {
            "W1": {
                "case_id": "edge64",
                "input_tokens": 64,
                "decode_tokens": 2,
            }
        }
        or queue.get("request_wall_limit_seconds") != 1800
    ):
        raise ValueError("Round12a frozen ABBA queue changed")
    queue["sha256"] = hashlib.sha256(raw_queue).hexdigest()
    runs = {
        version: validate_run(version, items[version], queue)
        for version in ROLES
    }
    blocks = {}
    for block, kt_version, nonkt_version in (
        ("A", 174, 175),
        ("B", 177, 176),
    ):
        kt_rows = items[kt_version]["final"]["results"]
        nonkt_rows = items[nonkt_version]["final"]["results"]
        metrics = {}
        for workload in queue["workloads"]:
            kt = [
                row
                for row in kt_rows
                if row["workload"] == workload and row["phase"] == "measurement"
            ]
            nonkt = [
                row
                for row in nonkt_rows
                if row["workload"] == workload
                and row["phase"] == "measurement"
            ]
            if len(kt) != 3 or len(nonkt) != 3:
                raise ValueError(f"block {block} {workload} sample count mismatch")
            row = {}
            for field in ("e2e_seconds", "ttft_seconds", "tpot_seconds"):
                kt_median = _median(kt, field)
                nonkt_median = _median(nonkt, field)
                row[field] = {
                    "kt_median": kt_median,
                    "nonkt_median": nonkt_median,
                    "kt_over_nonkt": kt_median / nonkt_median,
                    "kt_improvement_percent": (1 - kt_median / nonkt_median) * 100,
                }
            row["output_tokens_exact_across_modes"] = all(
                left["output_ids"] == right["output_ids"]
                for left, right in zip(kt, nonkt)
            )
            row["kt_e2e_at_least_5_percent_faster"] = (
                row["e2e_seconds"]["kt_over_nonkt"] <= 0.95
            )
            metrics[workload] = row
        blocks[block] = {
            "kt_version": kt_version,
            "nonkt_version": nonkt_version,
            "metrics": metrics,
            "pass_value_gate": all(
                row["kt_e2e_at_least_5_percent_faster"]
                for row in metrics.values()
            ),
        }
    value_pass = all(block["pass_value_gate"] for block in blocks.values())
    direction_consistent = all(
        row["e2e_seconds"]["kt_over_nonkt"] < 1
        for block in blocks.values()
        for row in block["metrics"].values()
    )
    if value_pass:
        status = "PASS_ROUND12A_KT_STYLE_VALUE_CONFIRMED"
        decision = "KT_STYLE_HETEROGENEOUS_EXECUTION_WORTH_IT_IN_QUALIFIED_SCOPE"
    elif direction_consistent:
        status = "ROUND12A_KT_FASTER_BELOW_PREREGISTERED_VALUE_MARGIN"
        decision = "KT_STYLE_VALUE_POSITIVE_BUT_MARGIN_NOT_QUALIFIED"
    else:
        status = "ROUND12A_KT_STYLE_VALUE_NOT_STABLY_SHOWN"
        decision = "NO_STABLE_KT_STYLE_REQUEST_VALUE_IN_TESTED_SCOPE"
    return {
        "schema": 1,
        "status": status,
        "decision": decision,
        "scope": "SERIAL_TP1_ASCEND12_KUNPENG_C128_PROMOTED_QSA",
        "fairness": {
            "same_model_revision": True,
            "same_promoted_qsa_tuple": True,
            "same_requests_and_sampling": True,
            "same_device_and_cpu_admission": True,
            "same_process_order": "KT/NON_KT/NON_KT/KT",
            "intrinsic_backend_precision_difference": (
                "KT routed experts use MXFP4 CPU bank with Static Q32 BF16/NZ29; "
                "non-KT OffloaderV1 stages full BF16 decoder layers"
            ),
            "claim_boundary": "SYSTEM_LEVEL_KT_STYLE_VS_SUPPORTED_SGLANG_OFFLOADERV1",
        },
        "profiler": "OFF",
        "timing_primary": "CLIENT_UNPROFILED_E2E",
        "value_gate": "KT/NON_KT E2E <= 0.95 in every block and queued workload",
        "queue": queue,
        "runs": runs,
        "blocks": blocks,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for version in ROLES:
        parser.add_argument(f"--v{version}", type=Path, required=True)
    parser.add_argument("--queue", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    items = {
        version: load_run(getattr(args, f"v{version}"), version)
        for version in ROLES
    }
    report = compare(items, args.queue)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(report["status"], report["decision"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
