"""Validate and summarize host-observed Hybrid timing without adding syncs."""

from __future__ import annotations

from collections import defaultdict
import hashlib
import json
from pathlib import Path
import statistics


STAGES = ("CPU_SUBMIT", "NPU_RESIDENT_APPLY", "CPU_RESULT", "ORIGINAL_HYBRID_APPLY")


def percentile(values, fraction):
    ordered = sorted(values)
    if not ordered:
        raise ValueError("empty percentile")
    return ordered[min(len(ordered) - 1, int((len(ordered) - 1) * fraction))]


def summarize(events):
    selections = [row for row in events if row.get("stage") == "SELECTION"]
    intervals = [row for row in events if row.get("stage") in STAGES]
    groups = defaultdict(dict)
    for row in intervals:
        key = (row["request_id"], row["generation"], row["start"], row["layer"])
        if row["stage"] in groups[key]:
            raise ValueError(f"duplicate timing stage: {key} {row['stage']}")
        if row.get("outcome") != "RETURN" or row["end_ns"] < row["start_ns"]:
            raise ValueError("invalid timing interval")
        groups[key][row["stage"]] = row
    if not groups or any(set(rows) != set(STAGES) for rows in groups.values()):
        raise ValueError("incomplete Hybrid timing group")
    selection_keys = {
        (row["request_id"], row["generation"], row["start"]): row for row in selections
    }
    values = defaultdict(list)
    token_values = defaultdict(lambda: defaultdict(list))
    for (request_id, generation, start, layer), rows in groups.items():
        selection = selection_keys[(request_id, generation, start)]
        if selection["path"] != "HYBRID" or selection["basis"] != "REQUEST_TOTAL":
            raise ValueError("unexpected Hybrid selection identity")
        outer = rows["ORIGINAL_HYBRID_APPLY"]
        for stage in STAGES[:-1]:
            inner = rows[stage]
            if not (outer["start_ns"] <= inner["start_ns"] <= inner["end_ns"] <= outer["end_ns"]):
                raise ValueError("stage is not enclosed by apply interval")
        for stage, row in rows.items():
            duration = (row["end_ns"] - row["start_ns"]) / 1e6
            values[(layer, stage)].append(duration)
            token_values[selection["batch_tokens"]][stage].append(duration)
    def stats(samples):
        return {
            "count": len(samples),
            "median_ms": statistics.median(samples),
            "p95_observed_ms": percentile(samples, 0.95),
            "min_ms": min(samples),
            "max_ms": max(samples),
        }
    return {
        "selection_count": len(selections),
        "timed_layer_apply_count": len(groups),
        "layers": sorted({key[0] for key in values}),
        "by_layer_stage": {
            f"layer{layer}.{stage}": stats(samples)
            for (layer, stage), samples in sorted(values.items())
        },
        "by_batch_tokens": {
            str(tokens): {stage: stats(samples) for stage, samples in sorted(stages.items())}
            for tokens, stages in sorted(token_values.items())
        },
    }


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    root = Path(__file__).resolve().parent
    evidence = root / "remote-evidence" / "v95"
    fixture = root / "t2-fixtures"
    events = [json.loads(line) for line in (evidence / "hybrid-path.jsonl").read_text().splitlines()]
    timing = summarize(events)
    result = json.loads((evidence / "round10-t1-results.json").read_text())
    receipt = json.loads((fixture / "real-prefill.receipt.json").read_text())
    if receipt["tensor_sha256"] != sha256(fixture / "real-prefill.pt"):
        raise ValueError("fixture tensor hash mismatch")
    h64 = [
        {"phase": row["phase"], "e2e_seconds": row["e2e_seconds"]}
        for row in result["results"]
        if row["case_id"] == "edge64"
    ]
    decode = [
        {
            "e2e_seconds": row["e2e_seconds"],
            "completion_tokens": row["validated"]["completion_tokens"],
            "seconds_per_completion_token_including_prefill":
                row["e2e_seconds"] / row["validated"]["completion_tokens"],
            "finish_reason": row["validated"]["finish_reason"],
        }
        for row in result["results"]
        if row["case_id"] == "decode64"
    ]
    document = {
        "schema": 1,
        "status": "PASS_T2_HYBRID_COST_AND_FIXTURE",
        "source_run": "v95-R",
        "instrumentation": {
            "kind": "host-only monotonic intervals",
            "added_device_synchronization": False,
            "timed_layers": timing["layers"],
            "limitations": [
                "CPU_SUBMIT and NPU_RESIDENT_APPLY measure host return intervals, not device completion",
                "CPU_RESULT is the exposed native result wait inside the enclosing apply interval",
                "nested intervals are not added to request E2E",
                "layers 0 and 3 are samples and are not multiplied across all layers",
                "decode has request-level E2E/completion evidence but no per-stage synchronized device timing",
            ],
        },
        "timing": timing,
        "request_cost": {"h64": h64, "decode": decode},
        "fixture": {
            "source": "qualified v80 real request capture reused read-only",
            "request": receipt["request"],
            "groups": receipt["groups"],
            "performance_evidence": False,
            "tensor_sha256": sha256(fixture / "real-prefill.pt"),
            "receipt_sha256": sha256(fixture / "real-prefill.receipt.json"),
            "route_tape_sha256": sha256(fixture / "real-prefill.routes.jsonl"),
        },
        "disposition": {
            "new_device_run_used": False,
            "critical_path_finding": "CPU_RESULT is an exposed blocking subset of Hybrid apply; submit and resident apply return intervals are much smaller host-side launch intervals",
            "next": "T3 native Kunpeng worker component verification",
        },
    }
    (root / "T2_COST_ATTRIBUTION.json").write_text(json.dumps(document, indent=2) + "\n")
    print(json.dumps({"status": document["status"], "timing": timing}, indent=2))


if __name__ == "__main__":
    main()
