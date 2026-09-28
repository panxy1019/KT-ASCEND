#!/usr/bin/env python3
"""Strictly combine the two fresh Round14a B S0 attribution blocks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics


COMPONENTS = (
    "GDN", "QSA_ATTENTION", "QSA_INDEXER", "PLE_PREPARE",
    "PLE_EXPOSED_WAIT", "PLE_CONSUMER", "ROUTED_EXPERT",
    "CPU_SUBMIT", "CPU_RESULT_WAIT", "NPU_RESIDENT",
)
EXPERT_ADDED_LAYER_BYTES = 480 * 3 * 2560 * 640 * 2

# These factors only normalize the deliberately sampled layer subsets to the
# number of providers present in the frozen Qwen3.8 graph.  The resulting
# values remain independent clock-domain projections: parent spans overlap and
# must never be summed into an estimated request wall time.
LAYER_COUNT_PROJECTION_FACTORS = {
    "Expert": 48 / 3,
    "KV+GDN state": 36 / 3,
    "Dense backbone QSA": 12 / 3,
    "PLE": 1,
}


def _read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line]


def _metric(span):
    return (float(span["device_ms"]) if span.get("device_status") == "COMPLETE"
            else float(span["duration_ns"]) / 1e6)


def _request_attribution(result, request):
    selected = [row for row in result["batches"]
                if row["request_id"] == request["request_id"]]
    phases = {}
    for phase in ("extend", "decode"):
        batches = [row for row in selected if row["mode"] == phase]
        values = {name: 0.0 for name in COMPONENTS}
        calls = {name: 0 for name in COMPONENTS}
        wall = 0.0
        for batch in batches:
            wall += (batch["ended_ns"] - batch["started_ns"]) / 1e6
            for span in batch["spans"]:
                values[span["component"]] += _metric(span)
                calls[span["component"]] += 1
        values["QSA_EXCLUSIVE"] = max(0.0, values["QSA_ATTENTION"] - values["QSA_INDEXER"])
        values["EXPERT_EXCLUSIVE_DEVICE"] = max(
            0.0, values["ROUTED_EXPERT"] - values["NPU_RESIDENT"])
        phases[phase] = {
            "phase_wall_ms": wall,
            "sampled_attributable_ms": values,
            "sampled_calls": calls,
            "fractions_of_phase_wall": {key: value / wall for key, value in values.items()},
            "five_bucket_time_ledger": {
                "Expert": {
                    "parent_device_sample_ms": values["ROUTED_EXPERT"],
                    "exclusive_device_sample_ms": values["EXPERT_EXCLUSIVE_DEVICE"],
                    "npu_resident_child_ms": values["NPU_RESIDENT"],
                    "cpu_submit_host_ms": values["CPU_SUBMIT"],
                    "cpu_result_exposed_wait_ms": values["CPU_RESULT_WAIT"],
                    "sampled_layers": [0, 24, 47],
                    "actual_layer_calls": "NOT_EXTRAPOLATED_ACROSS_OVERLAPPING_CLOCK_DOMAINS",
                },
                "PLE": {
                    "prepare_host_ms": values["PLE_PREPARE"],
                    "exposed_wait_host_ms": values["PLE_EXPOSED_WAIT"],
                    "consumer_device_ms": values["PLE_CONSUMER"],
                    "sampled_layers": [1],
                },
                "Dense backbone": {
                    "qsa_parent_device_sample_ms": values["QSA_ATTENTION"],
                    "qsa_indexer_child_device_ms": values["QSA_INDEXER"],
                    "qsa_exclusive_device_sample_ms": values["QSA_EXCLUSIVE"],
                    "sampled_layers": [3, 23, 47],
                    "remaining_dense_device_time": "UNKNOWN_NOT_SUMMED_FROM_PARENT_SPANS",
                },
                "KV+GDN state": {
                    "gdn_device_sample_ms": values["GDN"],
                    "sampled_layers": [0, 24, 46],
                },
                "Workspace": {"compute_ms": "NOT_AN_EXECUTION_PROVIDER"},
            },
        }
    return {"workload": request["workload"], "request_id": request["request_id"],
            "ttft_seconds": request["ttft_seconds"],
            "e2e_seconds": request["e2e_seconds"],
            "tpot_seconds": request["tpot_seconds"], "phases": phases}


def _resource(root, result):
    rows = _read_jsonl(root / "monitor.jsonl")
    sampled = [row for row in rows if "hbm_used_mib" in row]
    cgroup_peaks = []
    cgroup_currents = []
    for row in sampled:
        cg = row.get("owned_cgroup", {})
        if "memory.peak" in cg:
            cgroup_peaks.append(int(cg["memory.peak"]))
        elif "memory.max_usage_in_bytes" in cg:
            cgroup_peaks.append(int(cg["memory.max_usage_in_bytes"]))
        if "memory.current" in cg:
            cgroup_currents.append(int(cg["memory.current"]))
        elif "memory.usage_in_bytes" in cg:
            cgroup_currents.append(int(cg["memory.usage_in_bytes"]))
    board_peak = max(row["hbm_used_mib"] for row in sampled)
    baseline = max(row["system_baseline_mib"] for row in sampled
                   if row.get("system_baseline_mib") is not None)
    memory = result["five_bucket_memory"]
    return {
        "board_hbm_peak_mib": board_peak,
        "system_hbm_baseline_mib": baseline,
        "inferred_owned_board_increment_mib": board_peak - baseline,
        "host_cgroup_peak_bytes": max(cgroup_peaks),
        "host_cgroup_peak_gib": max(cgroup_peaks) / 2**30,
        "host_cgroup_sampled_current_peak_bytes": max(cgroup_currents),
        "allocator": memory["allocator"],
        "storage_buckets": memory["buckets"],
        "host_read_only_bank": memory["host_read_only_bank"],
        "declared_transient_reservations": memory["declared_transient_reservations"],
        "alias_count": len(memory["aliases"]),
    }


def _find(root):
    matches = list(Path(root).glob("20260928-round14a-b-profile-*"))
    if len(matches) != 1:
        raise RuntimeError(f"expected one extracted evidence root under {root}")
    return matches[0]


def _nonadditive_decode_projection(blocks):
    projected = []
    for block in blocks:
        request = next(row for row in block["requests"] if row["workload"] == "W2")
        decode = request["phases"]["decode"]["five_bucket_time_ledger"]
        values = {
            "Expert": decode["Expert"]["parent_device_sample_ms"]
                * LAYER_COUNT_PROJECTION_FACTORS["Expert"],
            "KV+GDN state": decode["KV+GDN state"]["gdn_device_sample_ms"]
                * LAYER_COUNT_PROJECTION_FACTORS["KV+GDN state"],
            "Dense backbone QSA": decode["Dense backbone"]["qsa_parent_device_sample_ms"]
                * LAYER_COUNT_PROJECTION_FACTORS["Dense backbone QSA"],
            "PLE": decode["PLE"]["consumer_device_ms"]
                * LAYER_COUNT_PROJECTION_FACTORS["PLE"],
        }
        projected.append({
            "version": block["version"],
            "block": block["block"],
            "workload": "W2",
            "phase": "decode",
            "layer_count_projection_factors": LAYER_COUNT_PROJECTION_FACTORS,
            "projected_parent_or_provider_ms": values,
            "rank_descending": [name for name, _ in sorted(
                values.items(), key=lambda item: item[1], reverse=True)],
            "interpretation": (
                "NONADDITIVE_LAYER_COUNT_NORMALIZATION_FOR_RANKING_ONLY; "
                "OVERLAPPING_PARENT_SPANS_AND_HOST_WAITS_MUST_NOT_BE_SUMMED"
            ),
        })
    return projected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v298", type=Path, required=True)
    parser.add_argument("--v299", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    blocks = []
    for version, expected_block, base in ((298, "A", args.v298), (299, "B", args.v299)):
        root = _find(base)
        result = json.loads((root / "round14a-b-profile-final.json").read_text())
        supervisor = json.loads((root / "round14a-b-profile-supervisor.json").read_text())
        if (result.get("status") != "PASS_ROUND14A_B_PROFILE_PROCESS"
                or supervisor.get("status") != result["status"]
                or result.get("version") != version or result.get("block") != expected_block
                or result.get("profile_threshold") != 129
                or result.get("generation_attempts") != 4
                or result.get("cleanup") != {"guard_exit_ack": True,
                                              "container_removed": True,
                                              "physical12_after": "EMPTY"}):
            raise RuntimeError(f"v{version} process/identity/cleanup contract failed")
        if [row["workload"] for row in result["measurement_requests"]] != ["W1", "W2"]:
            raise RuntimeError(f"v{version} request schedule mismatch")
        blocks.append({
            "version": version, "block": expected_block,
            "resources": _resource(root, result),
            "requests": [_request_attribution(result, row)
                         for row in result["measurement_requests"]],
        })

    memory_identities = [{key: row["resources"][key] for key in (
        "allocator", "storage_buckets", "declared_transient_reservations", "alias_count")}
        for row in blocks]
    memory_repeat_exact = memory_identities[0] == memory_identities[1]
    if not memory_repeat_exact:
        raise RuntimeError("fresh S0 five-bucket storage census is not repeat-exact")

    base_alloc_mib = blocks[0]["resources"]["allocator"]["allocated_bytes"] / 2**20
    base_owned_mib = max(row["resources"]["inferred_owned_board_increment_mib"]
                         for row in blocks)
    state_mib = blocks[0]["resources"]["storage_buckets"]["KV+GDN state"]["npu_bytes"] / 2**20
    workspace_mib = blocks[0]["resources"]["allocator"]["workspace_unclassified_allocated_bytes"] / 2**20
    reserve = {
        "promoted_gdn_workspace_reserve_mib": 1024.0,
        "c4_additional_kv_gdn_state_mib": 3 * state_mib,
        "c4_additional_workspace_mib": 3 * workspace_mib,
        "load_and_guard_margin_mib": 1024.0,
    }
    reserve_total = sum(reserve.values())
    caps = {"allocator_mib": 36352.0, "owned_mib": 37376.0,
            "board_mib": 56832.0}
    candidates = []
    for k in range(0, 7):
        added_mib = k * EXPERT_ADDED_LAYER_BYTES / 2**20
        candidates.append({
            "k": k, "added_expert_mib": added_mib,
            "projected_allocator_with_reserves_mib": base_alloc_mib + added_mib + reserve_total,
            "projected_owned_with_reserves_mib": base_owned_mib + added_mib + reserve_total,
            "projected_board_with_reserves_mib": max(
                row["resources"]["board_hbm_peak_mib"] for row in blocks) + added_mib + reserve_total,
        })
    safe = [row for row in candidates
            if row["projected_allocator_with_reserves_mib"] <= caps["allocator_mib"]
            and row["projected_owned_with_reserves_mib"] <= caps["owned_mib"]
            and row["projected_board_with_reserves_mib"] <= caps["board_mib"]]
    k_safe = max(row["k"] for row in safe)
    if k_safe != 1:
        raise RuntimeError(f"frozen conservative reserve arithmetic expected k_safe=1, got {k_safe}")

    output = {
        "schema": 1,
        "status": "PASS_ROUND14A_B_TWO_FRESH_S0_FIVE_BUCKET_ATTRIBUTION",
        "blocks": blocks,
        "memory_repeat_exact": True,
        "five_bucket_interpretation": {
            "memory_largest": "Expert",
            "decode_sampled_time_largest": "Expert",
            "expert_bottleneck": "CPU_MXFP4_DRAM_COMPUTE_PLUS_RESULT_DEPENDENCY",
            "layerwise_weight_transfer_in_hybrid": 0,
            "ple_exposed_wait_material": False,
            "gdn_and_qsa_location": "ALREADY_NPU",
            "transfer_bytes": {
                "expert_activation_and_result": "UNKNOWN_NO_RUNTIME_BYTE_COUNTER",
                "ple_h2d": "UNKNOWN_NO_RUNTIME_BYTE_COUNTER",
                "layerwise_expert_weights": 0,
            },
            "clock_domain_rule": "PARENT_DEVICE_SPANS_AND_HOST_WAITS_REPORTED_SEPARATELY_NOT_SUMMED",
            "w2_decode_nonadditive_layer_count_projections":
                _nonadditive_decode_projection(blocks),
        },
        "fplus_capacity": {
            "increment_per_full_front_layer_bytes": EXPERT_ADDED_LAYER_BYTES,
            "increment_per_full_front_layer_mib": EXPERT_ADDED_LAYER_BYTES / 2**20,
            "increment_per_full_front_layer_gib": EXPERT_ADDED_LAYER_BYTES / 2**30,
            "caps": caps,
            "baseline_allocator_mib": base_alloc_mib,
            "baseline_owned_mib": base_owned_mib,
            "reserves": reserve,
            "reserve_total_mib": reserve_total,
            "projections": candidates,
            "k_safe": k_safe,
            "profile": "FPLUS(1)_LAYER0_Q512_LAYERS1_47_Q32",
            "same_nominal_control": "U_PLUS_Q42_ALL48",
        },
        "conclusion": "SCREEN_FPLUS1_AND_UPLUS_Q42_THEN_PROMOTE_AT_MOST_ONE_TO_REQUEST_ABBA",
    }
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    print(output["status"], "K_SAFE", k_safe)


if __name__ == "__main__":
    main()
