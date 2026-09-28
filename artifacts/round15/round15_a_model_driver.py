#!/usr/bin/env python3
"""Run one bounded Round15 QSA full-set quality/performance screen."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

ROLES = {
    341: ("screen", "baseline", "S"),
    342: ("screen", "candidate", "S"),
}
QSA_LAYERS = list(range(3, 48, 4))
GDN_LAYERS = sorted(set(range(48)) - set(QSA_LAYERS))
WORKLOADS = {
    "W1": ("edge128", 16),
    "W2": ("edge128", 128),
}
CANARIES = (
    "canary_marker",
    "canary_math391",
    "canary_json",
    "canary_quote",
    "canary_transform",
)


def save(path, value):
    temporary = Path(path).with_suffix(".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def read_jsonl(path):
    path = Path(path)
    if not path.exists():
        return []
    raw = path.read_bytes()
    if raw and not raw.endswith(b"\n"):
        raise RuntimeError(f"unterminated evidence file {path.name}")
    return [json.loads(line) for line in raw.splitlines()]


def request(rid, ids, tokens, *, stream=False, ignore_eos=False):
    return {
        "rid": rid,
        "input_ids": list(ids),
        "stream": stream,
        "sampling_params": {
            "temperature": 0,
            "top_p": 1,
            "sampling_seed": 20260920,
            "max_new_tokens": tokens,
            "ignore_eos": ignore_eos,
        },
        "return_logprob": True,
        "logprob_start_len": -1,
    }


def _quality_row(fixture, version, phase, case_id, tokens=32, *, stream=False, ignore_eos=False, validator=None):
    case = fixture["cases"][case_id]
    return {
        "kind": "quality",
        "case_id": case_id,
        "phase": phase,
        "validator": validator or case["validator"],
        "input_tokens": len(case["input_ids"]),
        "decode_tokens": tokens,
        "request": request(
            f"r15-a-v{version}-{phase}-{case_id}",
            case["input_ids"],
            tokens,
            stream=stream,
            ignore_eos=ignore_eos,
        ),
    }


def build_schedule(fixture10, version):
    stage, implementation, block = ROLES[version]
    rows = [
        _quality_row(fixture10, version, "quality", case_id)
        for case_id in CANARIES
    ]
    rows.append(_quality_row(
        fixture10, version, "multi_step", "edge128", tokens=32,
        ignore_eos=True, validator={"kind": "multi_step", "minimum_tokens": 32},
    ))
    rows.append(_quality_row(
        fixture10, version, "cancel", "cancel128", tokens=128,
        stream=True, ignore_eos=True, validator={"kind": "abort"},
    ))
    rows.append(_quality_row(fixture10, version, "reuse", "canary_math391"))
    for workload, (case_id, tokens) in WORKLOADS.items():
        case = fixture10["cases"][case_id]
        for phase, repeat in (
            ("warmup", 0),
            ("measurement", 1),
            ("measurement", 2),
        ):
            rows.append({
                "kind": "performance",
                "workload": workload,
                "case_id": case_id,
                "phase": phase,
                "repeat": repeat,
                "input_tokens": len(case["input_ids"]),
                "decode_tokens": tokens,
                "request": request(
                    f"r15-a-v{version}-{implementation}-{block.lower()}-{workload.lower()}-{phase}-{repeat}",
                    case["input_ids"], tokens, stream=True, ignore_eos=True,
                ),
            })
    return rows


def validate_qsa(request_id, events):
    rows = [
        row for row in events
        if row.get("status") == "CONTEXT_REFERENCE_QSA_DISPATCH"
        and row.get("request_id") == request_id
    ]
    vector = sorted({row.get("layer") for row in rows if row.get("strategy") == "VECTORIZED"})
    direct = sorted({row.get("layer") for row in rows if row.get("strategy") == "DIRECT_BF16_ROW"})
    unexpected = sorted({
        row.get("strategy") for row in rows
        if row.get("strategy") not in {"VECTORIZED", "DIRECT_BF16_ROW"}
    })
    if vector != QSA_LAYERS or direct != QSA_LAYERS or unexpected:
        raise RuntimeError(
            f"Round14a QSA provider identity missing: vector={vector}, direct={direct}, unexpected={unexpected}"
        )
    return {
        "prefill_strategy": "VECTORIZED",
        "decode_strategy": "DIRECT_BF16_ROW",
        "layers": QSA_LAYERS,
        "dispatch_count": len(rows),
    }


def validate_ple(request_id, events):
    if not any(row.get("status") == "PLE_ASYNC_BOUND" for row in events):
        raise RuntimeError("PLE ASYNC binding missing")
    rows = [
        row for row in events
        if row.get("status") == "PLE_BATCH_DRAINED"
        and row.get("request_id") == request_id
    ]
    if not rows or not any(row.get("async_engaged") is True for row in rows):
        raise RuntimeError("request did not engage and drain PLE ASYNC")
    return {"drained": len(rows), "async": sum(row.get("async_engaged") is True for row in rows)}


def validate_gdn_bindings(path):
    rows = read_jsonl(path)
    selected = [row for row in rows if row.get("status") == "ROUND14A_GDN_BRANCH_HIT"]
    layers = sorted({row.get("layer") for row in selected})
    branches = {row.get("implementation") for row in selected}
    if layers != GDN_LAYERS or branches != {"candidate"} or len(selected) != len(GDN_LAYERS):
        raise RuntimeError(
            f"actual GDN branch evidence mismatch: layers={layers}, branches={branches}, rows={len(selected)}"
        )
    return {"implementation": "candidate", "layers": layers, "first_hit_rows": len(selected)}


def validate_selection_binding(path, implementation):
    rows = read_jsonl(path)
    selected = [
        row for row in rows
        if row.get("status") == "ROUND15_QSA_FULL_SET_HIT"
    ]
    if implementation == "baseline":
        if selected:
            raise RuntimeError("baseline unexpectedly entered the Round15 QSA candidate")
        return {"implementation": "EAGER_REFERENCE", "layers": [], "first_hit_rows": 0}
    layers = sorted({row.get("layer") for row in selected})
    strategies = {row.get("implementation") for row in selected}
    if layers != QSA_LAYERS or strategies != {"FULL_SET_CAUSAL_SHORT_CONTEXT"} or len(selected) != len(QSA_LAYERS):
        raise RuntimeError(
            f"Round15 QSA selection binding mismatch: layers={layers}, strategies={strategies}, rows={len(selected)}"
        )
    return {
        "implementation": "FULL_SET_CAUSAL_SHORT_CONTEXT",
        "layers": layers,
        "first_hit_rows": len(selected),
    }


def run(version, out, fixture10_path):
    from round11_t4_driver import consume_stream, post, validate_hybrid_path
    from round10_t1_driver import run_cancel
    from round10_t1_protocol import validate_cancel_receipt, validate_response

    stage, implementation, block = ROLES[version]
    out = Path(out)
    result_path = out / "round15-a-model-raw.json"
    if result_path.exists():
        raise FileExistsError(result_path)
    fixture10 = json.loads(Path(fixture10_path).read_text())
    schedule = build_schedule(fixture10, version)
    expected_attempts = 14
    expected_aborts = 1
    report = {
        "schema": 1,
        "status": "RUNNING",
        "stage": f"ROUND15_A_QSA_{stage.upper()}",
        "version": version,
        "implementation": implementation,
        "block": block,
        "profiler": "OFF",
        "threshold": 129,
        "placement": "UNIFORM_Q32",
        "dynamic": "OFF",
        "ple": "ASYNC",
        "qsa_prefill": "VECTORIZED",
        "qsa_decode": "DIRECT_BF16_ROW",
        "generation_budget": expected_attempts,
        "generation_attempts": 0,
        "abort_controls": 0,
        "automatic_retries": 0,
        "results": [],
        "measurement_windows": {},
    }
    pending_cancel = None
    try:
        for row in schedule:
            report["generation_attempts"] += 1
            rid = row["request"]["rid"]
            started = time.monotonic_ns()
            if row["phase"] == "cancel":
                report["abort_controls"] += 1
                pending_cancel = run_cancel(row, out, post_fn=post)
                report["results"].append({
                    "case_id": row["case_id"],
                    "phase": "cancel",
                    "request_id": rid,
                    "e2e_seconds": (time.monotonic_ns() - started) / 1e9,
                    "cancel_status": pending_cancel["status"],
                })
                save(result_path, report)
                continue
            if row["kind"] == "performance":
                with post("/generate", row["request"], 1800) as response:
                    stream = consume_stream(response, row["decode_tokens"])
                ended = time.monotonic_ns()
                arrivals = stream["token_arrival_monotonic_ns"]
                if stream["output_ids"][:2] != [3897, 248046]:
                    raise RuntimeError("performance request lost frozen ACK prefix")
                if stream["completion_tokens"] != row["decode_tokens"]:
                    raise RuntimeError("performance request did not produce fixed output budget")
                item = {
                    **{key: row[key] for key in (
                        "workload", "case_id", "phase", "repeat", "input_tokens", "decode_tokens"
                    )},
                    "request_id": rid,
                    "completion_tokens": stream["completion_tokens"],
                    "output_ids": stream["output_ids"],
                    "finite_logprobs": stream["finite_logprobs"],
                    "finish_reason": stream["finish_reason"],
                    "ttft_seconds": (arrivals[0] - started) / 1e9,
                    "e2e_seconds": (ended - started) / 1e9,
                    "tpot_seconds": (
                        (arrivals[-1] - arrivals[0]) / (len(arrivals) - 1) / 1e9
                        if len(arrivals) > 1 else None
                    ),
                    "path": validate_hybrid_path(
                        rid, read_jsonl(out / "hybrid-path.jsonl"),
                        read_jsonl(out / "layerwise-events.jsonl"),
                    ),
                    "qsa": validate_qsa(
                        rid, read_jsonl(out / "context-reference-events.jsonl")
                    ),
                    "ple": validate_ple(rid, read_jsonl(out / "ple-async-events.jsonl")),
                }
                report["results"].append(item)
                save(result_path, report)
                continue

            with post("/generate", row["request"], 1800) as response:
                value = json.load(response)
            validated = validate_response(row, value)
            item = {
                "case_id": row["case_id"],
                "phase": row["phase"],
                "request_id": rid,
                "e2e_seconds": (time.monotonic_ns() - started) / 1e9,
                "quality_pass": True,
                "validated": validated,
                "path": validate_hybrid_path(
                    rid, read_jsonl(out / "hybrid-path.jsonl"),
                    read_jsonl(out / "layerwise-events.jsonl"),
                ),
                "qsa": validate_qsa(
                    rid, read_jsonl(out / "context-reference-events.jsonl")
                ),
                "ple": validate_ple(rid, read_jsonl(out / "ple-async-events.jsonl")),
            }
            report["results"].append(item)
            if row["phase"] == "reuse":
                if pending_cancel is None:
                    raise RuntimeError("reuse lacks preceding cancel")
                pending_cancel["reuse"] = {
                    "request_id": rid,
                    "validator_passed": True,
                    "output_ids": validated["output_ids"],
                }
                pending_cancel["status"] = "PASS_HYBRID_MIDPREFILL_CANCEL_AND_REUSE"
                validate_cancel_receipt(pending_cancel)
                save(out / "round15-a-cancel-receipt.json", pending_cancel)
                report["results"][-2]["cancel_status"] = pending_cancel["status"]
                pending_cancel = None
            save(result_path, report)

        for workload in WORKLOADS:
            selected = [
                row for row in report["results"]
                if row.get("workload") == workload and row.get("phase") == "measurement"
            ]
            if len(selected) != 2:
                raise RuntimeError(f"{workload} measurement count mismatch")
            first = next(
                index for index, row in enumerate(schedule)
                if row.get("workload") == workload and row.get("phase") == "measurement"
            )
            last = max(
                index for index, row in enumerate(schedule)
                if row.get("workload") == workload and row.get("phase") == "measurement"
            )
            seconds = sum(row["e2e_seconds"] for row in selected)
            tokens = sum(row["completion_tokens"] for row in selected)
            report["measurement_windows"][workload] = {
                "schedule_first_index": first,
                "schedule_last_index": last,
                "seconds": seconds,
                "successful_requests": 2,
                "output_tokens": tokens,
                "serial_output_tokens_per_second": tokens / seconds,
                "requests_per_second": 2 / seconds,
            }
        if (
            report["generation_attempts"] != expected_attempts
            or report["abort_controls"] != expected_aborts
            or pending_cancel is not None
        ):
            raise RuntimeError("Round15 A QSA schedule/budget incomplete")
        report["gdn_binding"] = validate_gdn_bindings(
            out / "round14a-gdn-branch.jsonl"
        )
        report["qsa_selection_binding"] = validate_selection_binding(
            out / "round15-qsa-selection.jsonl", implementation
        )
        report["status"] = "PASS_ROUND15_A_QSA_MODEL_DRIVER_RAW"
        save(result_path, report)
        return report
    except BaseException as error:
        report.update(
            status="FAIL_ROUND15_A_QSA_MODEL_DRIVER",
            error=f"{type(error).__name__}: {error}",
        )
        save(result_path, report)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", type=int, choices=tuple(ROLES), required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--fixture10", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.version, args.out, args.fixture10)
    print(result["status"], args.version, flush=True)


if __name__ == "__main__":
    main()
