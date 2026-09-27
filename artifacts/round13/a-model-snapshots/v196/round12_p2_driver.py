#!/usr/bin/env python3
"""Run one bounded Round12 baseline/candidate model smoke process."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import time


ROLE_VERSIONS = {153: "BASELINE", 154: "CANDIDATE"}
QSA_LAYERS = list(range(3, 48, 4))


def _save(path, value):
    temporary = Path(path).with_suffix(".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def _request(rid, ids, maximum, *, stream=False, ignore_eos=False):
    return {
        "rid": rid,
        "input_ids": list(ids),
        "stream": stream,
        "sampling_params": {
            "temperature": 0, "top_p": 1, "sampling_seed": 20260920,
            "max_new_tokens": maximum, "ignore_eos": ignore_eos,
        },
        "return_logprob": True,
        "logprob_start_len": -1,
    }


def build_schedule(fixture, *, version, mode):
    if ROLE_VERSIONS.get(version) != mode:
        raise ValueError("Round12 P2 role/version mismatch")
    definitions = (
        ("canary_math391", "warmup", 32, None),
        ("canary_marker", "canary", 32, None),
        ("canary_json", "canary", 32, None),
        ("canary_quote", "canary", 32, None),
        ("canary_transform", "canary", 32, None),
        ("edge128", "decode", 32, 32),
        ("cancel128", "cancel", 8, None),
        ("canary_math391", "reuse", 32, None),
    )
    rows = []
    for index, (case_id, phase, maximum, decode_tokens) in enumerate(definitions, 1):
        case = fixture["cases"][case_id]
        rid = f"r12-p2-v{version}-{mode.lower()}-{index:02d}-{case_id}-{phase}"
        rows.append({
            "index": index,
            "case_id": case_id,
            "phase": phase,
            "decode_tokens": decode_tokens,
            "validator": copy.deepcopy(case["validator"]),
            "request": _request(
                rid, case["input_ids"], maximum,
                stream=phase in {"decode", "cancel"},
                ignore_eos=phase == "decode"),
        })
    return rows


def _read_jsonl(path):
    path = Path(path)
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def validate_qsa_dispatch(request_id, events, *, mode, require_decode_row):
    rows = [row for row in events
            if row.get("status") == "CONTEXT_REFERENCE_QSA_DISPATCH"
            and row.get("request_id") == request_id]
    if mode == "BASELINE":
        if rows:
            raise RuntimeError("baseline unexpectedly emitted candidate QSA dispatch")
        return {"mode": "QUALIFIED_ORIGINAL", "dispatch_count": 0}
    if mode != "CANDIDATE":
        raise ValueError("unknown Round12 P2 mode")
    vector = [row for row in rows if row.get("strategy") == "VECTORIZED"]
    row_path = [row for row in rows if row.get("strategy") == "ROW"]
    vector_layers = sorted(set(row.get("layer") for row in vector))
    row_layers = sorted(set(row.get("layer") for row in row_path))
    if vector_layers != QSA_LAYERS:
        raise RuntimeError("candidate request did not hit all VECTOR QSA layers")
    if require_decode_row and row_layers != QSA_LAYERS:
        raise RuntimeError("candidate Decode did not retain all ROW QSA layers")
    if any(row.get("query_tokens", 0) <= 1 for row in vector):
        raise RuntimeError("candidate VECTOR dispatch has invalid query size")
    return {
        "mode": "APPROVED_CANDIDATE",
        "dispatch_count": len(rows),
        "vectorized_count": len(vector),
        "row_count": len(row_path),
        "vectorized_layers": vector_layers,
        "row_layers": row_layers,
    }


def validate_binding(events, mode):
    bindings = [row for row in events
                if row.get("status") == "CONTEXT_REFERENCE_BOUND"]
    if len(bindings) != 1:
        raise RuntimeError("QSA model process requires one context binding event")
    actual = bindings[0].get("qsa_source_binding", {}).get("mode")
    expected = "APPROVED_CANDIDATE" if mode == "CANDIDATE" else "QUALIFIED_ORIGINAL"
    if actual != expected:
        raise RuntimeError(f"QSA source binding mode mismatch: {actual} != {expected}")
    return bindings[0]["qsa_source_binding"]


def run(mode, version, out, fixture_path, *, post_fn=None):
    import round10_t1_driver as t1
    import round10_t1_protocol as protocol
    from round11_t4_driver import consume_stream

    if post_fn is None:
        post_fn = t1.post
    out = Path(out)
    result_path = out / "round12-p2-raw.json"
    if result_path.exists():
        raise FileExistsError(result_path)
    fixture = json.loads(Path(fixture_path).read_text())
    protocol.validate_fixture(fixture)
    schedule = build_schedule(fixture, version=version, mode=mode)
    report = {
        "schema": 1, "status": "RUNNING", "stage": "P2_SMOKE",
        "mode": mode, "version": version, "profiler": "OFF",
        "generation_budget": 8, "generation_attempts": 0,
        "abort_controls": 0, "automatic_retries": 0,
        "results": [], "cancel_receipts": [],
    }
    pending_cancel = None
    try:
        for row in schedule:
            report["generation_attempts"] += 1
            started = time.monotonic_ns()
            if row["phase"] == "cancel":
                report["abort_controls"] += 1
                pending_cancel = t1.run_cancel(row, out, post_fn=post_fn)
                report["results"].append({
                    "index": row["index"], "case_id": row["case_id"],
                    "phase": row["phase"], "request_id": row["request"]["rid"],
                    "e2e_seconds": (time.monotonic_ns() - started) / 1e9,
                    "cancel_status": pending_cancel["status"],
                })
                _save(result_path, report)
                continue
            if row["phase"] == "decode":
                with post_fn("/generate", row["request"], 1800) as response:
                    stream = consume_stream(response, row["decode_tokens"])
                if stream["output_ids"][:2] != [3897, 248046]:
                    raise RuntimeError("P2 Decode changed the frozen ACK prefix")
                result = {
                    "index": row["index"], "case_id": row["case_id"],
                    "phase": row["phase"], "request_id": row["request"]["rid"],
                    "input_tokens": len(row["request"]["input_ids"]),
                    "decode_tokens": row["decode_tokens"],
                    "e2e_seconds": (time.monotonic_ns() - started) / 1e9,
                    "quality_pass": True, **stream,
                }
            else:
                with post_fn("/generate", row["request"], 1800) as response:
                    value = json.load(response)
                validated = protocol.validate_response(row, value)
                result = {
                    "index": row["index"], "case_id": row["case_id"],
                    "phase": row["phase"], "request_id": row["request"]["rid"],
                    "e2e_seconds": (time.monotonic_ns() - started) / 1e9,
                    "validated": validated, "output_ids": validated["output_ids"],
                    "quality_pass": True,
                }
                if row["phase"] == "reuse":
                    if pending_cancel is None:
                        raise RuntimeError("P2 reuse is missing its cancellation")
                    pending_cancel["reuse"] = {
                        "request_id": row["request"]["rid"],
                        "validator_passed": True,
                        "output_ids": validated["output_ids"],
                    }
                    pending_cancel["status"] = "PASS_HYBRID_MIDPREFILL_CANCEL_AND_REUSE"
                    protocol.validate_cancel_receipt(pending_cancel)
                    receipt = out / "round12-p2-cancel-receipt.json"
                    _save(receipt, pending_cancel)
                    report["cancel_receipts"].append(receipt.name)
                    report["results"][-1]["cancel_status"] = pending_cancel["status"]
                    pending_cancel = None
            result["qsa_dispatch"] = validate_qsa_dispatch(
                row["request"]["rid"],
                _read_jsonl(out / "context-reference-events.jsonl"),
                mode=mode,
                require_decode_row=(row["phase"] == "decode"),
            )
            report["results"].append(result)
            _save(result_path, report)
        events = _read_jsonl(out / "context-reference-events.jsonl")
        report["qsa_source_binding"] = validate_binding(events, mode)
        if (report["generation_attempts"] != 8 or len(report["results"]) != 8
                or report["abort_controls"] != 1 or pending_cancel is not None
                or len(report["cancel_receipts"]) != 1):
            raise RuntimeError("Round12 P2 schedule is incomplete")
        report["status"] = "PASS_ROUND12_P2_DRIVER_RAW"
        _save(result_path, report)
        return report
    except BaseException as error:
        report.update(status="FAIL_ROUND12_P2_DRIVER",
                      error=f"{type(error).__name__}: {error}")
        _save(result_path, report)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("BASELINE", "CANDIDATE"), required=True)
    parser.add_argument("--version", type=int, choices=tuple(ROLE_VERSIONS), required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--fixture", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.mode, args.version, args.out, args.fixture)
    print(result["status"], args.mode, flush=True)


if __name__ == "__main__":
    main()
