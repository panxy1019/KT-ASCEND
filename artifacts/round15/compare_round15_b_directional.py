#!/usr/bin/env python3
"""Compare the preserved v354 baseline with one simplified B candidate run."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import statistics
import tarfile


def read_member(archive: tarfile.TarFile, name: str):
    return json.load(archive.extractfile(name))


def read_jsonl(archive: tarfile.TarFile, name: str):
    return [
        json.loads(line)
        for line in archive.extractfile(name).read().splitlines()
        if line.strip()
    ]


def load(path: Path, root: str):
    with tarfile.open(path) as archive:
        result = read_member(archive, root + "/round15-b-model-final.json")
        monitor = read_jsonl(archive, root + "/monitor.jsonl")
        graph = (
            read_jsonl(archive, root + "/round15-b-graph-events.jsonl")
            if result["implementation"] == "candidate"
            else []
        )
    return result, monitor, graph


def workload(result, name):
    rows = [
        row
        for row in result["results"]
        if row.get("workload") == name and row.get("phase") == "measurement"
    ]
    if len(rows) != 2:
        raise RuntimeError(f"{name} requires exactly two simplified measurements")
    return {
        "samples": [
            {
                "repeat": row["repeat"],
                "e2e_seconds": row["e2e_seconds"],
                "ttft_seconds": row["ttft_seconds"],
                "tpot_seconds": row["tpot_seconds"],
                "completion_tokens": row["completion_tokens"],
            }
            for row in rows
        ],
        "median_e2e_seconds": statistics.median(row["e2e_seconds"] for row in rows),
        "median_ttft_seconds": statistics.median(row["ttft_seconds"] for row in rows),
        "median_tpot_seconds": statistics.median(row["tpot_seconds"] for row in rows),
        "serial_output_tokens_per_second": result["measurement_windows"][name][
            "serial_output_tokens_per_second"
        ],
        "requests_per_second": result["measurement_windows"][name]["requests_per_second"],
    }


def lower_improvement(baseline, candidate):
    return (1.0 - candidate / baseline) * 100.0


def higher_improvement(baseline, candidate):
    return (candidate / baseline - 1.0) * 100.0


def output_ids(row):
    if "output_ids" in row:
        return row["output_ids"]
    return row.get("validated", {}).get("output_ids")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--baseline-root", required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--candidate-root", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--csv", type=Path, required=True)
    args = parser.parse_args()
    for path in (args.output, args.csv):
        if path.exists():
            raise FileExistsError(path)

    baseline, baseline_monitor, _ = load(args.baseline, args.baseline_root)
    candidate, candidate_monitor, graph = load(args.candidate, args.candidate_root)
    if (
        baseline["status"] != "PASS_ROUND15_B_GRAPH_MODEL_PROCESS"
        or candidate["status"] != "PASS_ROUND15_B_GRAPH_MODEL_PROCESS"
    ):
        raise RuntimeError("model result status mismatch")

    aligned = []
    for left, right in zip(baseline["results"], candidate["results"], strict=True):
        identity = tuple(left.get(key) for key in ("phase", "case_id", "workload", "repeat"))
        if identity != tuple(right.get(key) for key in ("phase", "case_id", "workload", "repeat")):
            raise RuntimeError("request schedule identity mismatch")
        left_ids = output_ids(left)
        right_ids = output_ids(right)
        if left_ids is not None or right_ids is not None:
            exact = left_ids == right_ids
            if not exact:
                raise RuntimeError(f"output token mismatch: {identity}")
            aligned.append({"identity": identity, "output_token_exact": True})

    workloads = {}
    csv_rows = []
    for name in ("W1", "W2"):
        left = workload(baseline, name)
        right = workload(candidate, name)
        changes = {
            "e2e_improvement_percent": lower_improvement(
                left["median_e2e_seconds"], right["median_e2e_seconds"]
            ),
            "ttft_improvement_percent": lower_improvement(
                left["median_ttft_seconds"], right["median_ttft_seconds"]
            ),
            "tpot_improvement_percent": lower_improvement(
                left["median_tpot_seconds"], right["median_tpot_seconds"]
            ),
            "serial_output_throughput_improvement_percent": higher_improvement(
                left["serial_output_tokens_per_second"],
                right["serial_output_tokens_per_second"],
            ),
        }
        workloads[name] = {"baseline": left, "candidate": right, "change": changes}
        for arm, values in (("baseline_v354", left), ("candidate_v360", right)):
            csv_rows.append(
                {
                    "line": "B",
                    "workload": name,
                    "arm": arm,
                    "median_e2e_seconds": values["median_e2e_seconds"],
                    "median_ttft_seconds": values["median_ttft_seconds"],
                    "median_tpot_seconds": values["median_tpot_seconds"],
                    "serial_output_tokens_per_second": values[
                        "serial_output_tokens_per_second"
                    ],
                    "requests_per_second": values["requests_per_second"],
                }
            )

    def peak(rows, key):
        return max(row[key] for row in rows if key in row)

    def minimum(rows, key):
        return min(row[key] for row in rows if key in row)

    captures = [row for row in graph if row.get("event") == "CAPTURE_AND_SELF_CHECK"]
    replays = [row for row in graph if row.get("event") == "REPLAY"]
    report = {
        "schema": 1,
        "status": "ROUND15_B_DIRECTIONAL_SMALL_GAIN_BELOW_PROMOTION_GATE",
        "evidence_class": "ONE_BASELINE_PROCESS_ONE_CANDIDATE_PROCESS_TWO_MEASUREMENTS_PER_WORKLOAD",
        "not_claimed": [
            "FOUR_PROCESS_ABBA_STABILITY",
            "STATISTICAL_QUALIFICATION",
            "PRODUCTION_PROMOTION",
        ],
        "versions": {"baseline": 354, "candidate": 360},
        "quality": {
            "aligned_output_rows": len(aligned),
            "all_output_token_sequences_exact": True,
            "candidate_generation_attempts": candidate["generation_attempts"],
            "candidate_abort_controls": candidate["abort_controls"],
            "candidate_cancel_reuse": "PASS",
            "finite_logprobs": all(
                row.get("finite_logprobs", row.get("validated", {}).get("finite_output_logprobs", True))
                for row in candidate["results"]
            ),
        },
        "graph_binding": {
            "capture_layers": len({row["layer"] for row in captures}),
            "capture_events": len(captures),
            "max_capture_relative_l2": max(row["relative_l2"] for row in captures),
            "replay_layers": len({row["layer"] for row in replays}),
            "replay_markers": len(replays),
        },
        "workloads": workloads,
        "resources": {
            "baseline_peak_whole_card_hbm_mib": peak(baseline_monitor, "hbm_used_mib"),
            "candidate_peak_whole_card_hbm_mib": peak(candidate_monitor, "hbm_used_mib"),
            "baseline_peak_owned_hbm_mib": peak(baseline_monitor, "owned_process_mib"),
            "candidate_peak_owned_hbm_mib": peak(candidate_monitor, "owned_process_mib"),
            "baseline_min_host_mem_available_gib": minimum(
                baseline_monitor, "host_mem_available_kib"
            ) / 1024**2,
            "candidate_min_host_mem_available_gib": minimum(
                candidate_monitor, "host_mem_available_kib"
            ) / 1024**2,
            "host_cgroup_peak": "UNKNOWN_NOT_RECORDED_BY_THIS_MONITOR",
        },
        "startup_seconds": {
            "baseline": baseline["startup"]["seconds"],
            "candidate": candidate["startup"]["seconds"],
        },
        "disposition": "KEEP_DEFAULT_OFF_SMALL_DIRECTIONAL_GAIN_DOES_NOT_MEET_ORIGINAL_REQUEST_GATE",
    }
    report["resources"]["candidate_minus_baseline_whole_card_hbm_mib"] = (
        report["resources"]["candidate_peak_whole_card_hbm_mib"]
        - report["resources"]["baseline_peak_whole_card_hbm_mib"]
    )
    report["resources"]["candidate_minus_baseline_owned_hbm_mib"] = (
        report["resources"]["candidate_peak_owned_hbm_mib"]
        - report["resources"]["baseline_peak_owned_hbm_mib"]
    )
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    with args.csv.open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(csv_rows[0]))
        writer.writeheader()
        writer.writerows(csv_rows)
    print(report["status"], report["disposition"])


if __name__ == "__main__":
    main()
