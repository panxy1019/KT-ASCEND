#!/usr/bin/env python3
"""Check bitwise logit equality only for requests with no CPU route anywhere."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def canonical_sha256(payload: dict) -> str:
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def request_routes(manifest: dict, trace_path: Path, preflight_manifest: dict) -> list[dict]:
    records = [json.loads(line) for line in trace_path.read_text().splitlines()]
    if len(preflight_manifest["rows"]) != 1:
        raise SystemExit(f"expected one preflight row: {trace_path}")
    skip = int(preflight_manifest["rows"][0]["deferred_trace_record_count"])
    expected = skip + sum(int(row["deferred_trace_record_count"]) for row in manifest["rows"])
    if len(records) != expected:
        raise SystemExit(f"trace record count mismatch for {trace_path}: {len(records)} != {expected}")
    position = skip
    result = []
    for row in manifest["rows"]:
        count = int(row["deferred_trace_record_count"])
        group = records[position:position + count]
        position += count
        submits = [record for record in group if record["event"] == "submit"]
        pending = [record for record in group if record["event"] == "pending_state"]
        if not submits or not pending or pending[-1]["pending_layers"]:
            raise SystemExit(f"missing submit or pending drain: {row['prompt_id']}:{row['token_index']}")
        result.append({
            "prompt_id": row["prompt_id"],
            "token_index": row["token_index"],
            "full_request_cpu_routes": sum(
                int(record["immediate_cpu_routes"]) + int(record["deferred_cpu_routes"])
                for record in submits
            ),
            "target_row_cpu_hits": int(row["cpu_hit_count"]),
            "submit_count": len(submits),
        })
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-manifest", type=Path, required=True)
    parser.add_argument("--baseline-trace", type=Path, required=True)
    parser.add_argument("--baseline-preflight", type=Path, required=True)
    parser.add_argument("--deferred-manifest", type=Path, required=True)
    parser.add_argument("--deferred-trace", type=Path, required=True)
    parser.add_argument("--deferred-preflight", type=Path, required=True)
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    baseline = load_json(args.baseline_manifest)
    deferred = load_json(args.deferred_manifest)
    metrics = load_json(args.metrics)
    if not (
        baseline["corpus_sha256"] == deferred["corpus_sha256"] == metrics["corpus_sha256"]
        and baseline["baseline_history_sha256"] == deferred["baseline_history_sha256"] == metrics["baseline_history_sha256"]
        and baseline["row_count"] == deferred["row_count"] == metrics["position_count"]
    ):
        raise SystemExit("corpus, history or position-count mismatch")
    baseline_routes = request_routes(baseline, args.baseline_trace, load_json(args.baseline_preflight))
    deferred_routes = request_routes(deferred, args.deferred_trace, load_json(args.deferred_preflight))
    metric_rows = metrics["rows"]
    if len(baseline_routes) != len(deferred_routes) or len(metric_rows) != len(baseline_routes):
        raise SystemExit("route and metric length mismatch")

    controls = []
    target_only_misses = 0
    failures = []
    for base_route, deferred_route, metric in zip(baseline_routes, deferred_routes, metric_rows):
        key = (metric["prompt_id"], metric["token_index"])
        if key != (base_route["prompt_id"], base_route["token_index"]) or key != (deferred_route["prompt_id"], deferred_route["token_index"]):
            raise SystemExit(f"route/metric position mismatch: {key}")
        if deferred_route["target_row_cpu_hits"] == 0 and deferred_route["full_request_cpu_routes"] > 0:
            target_only_misses += 1
        if base_route["full_request_cpu_routes"] == deferred_route["full_request_cpu_routes"] == 0:
            exact = metric["full_logit_max_abs"] == 0.0
            item = {"prompt_id": key[0], "token_index": key[1], "exact": exact,
                    "full_logit_max_abs": metric["full_logit_max_abs"]}
            controls.append(item)
            if not exact:
                failures.append(item)

    report = {
        "schema_version": 1,
        "control_definition": "zero CPU routes across every submit row in both full requests",
        "corpus_sha256": metrics["corpus_sha256"],
        "baseline_history_sha256": metrics["baseline_history_sha256"],
        "metrics_sha256": metrics["sha256"],
        "position_count": len(metric_rows),
        "full_request_cpu_not_hit_count": len(controls),
        "full_request_cpu_not_hit_exact_count": len(controls) - len(failures),
        "target_row_misses_with_prefix_cpu_routes": target_only_misses,
        "failures": failures,
        "status": "FAIL" if failures else "PASS" if controls else "NOT_OBSERVED",
    }
    report["sha256"] = canonical_sha256(report)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    if report["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
