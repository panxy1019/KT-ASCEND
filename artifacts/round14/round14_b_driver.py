#!/usr/bin/env python3
"""Run one Round14 B real-distribution H-BOUNDED qualification block."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import time
import urllib.error


ROLES = {
    243: ("H512", "A"),
    244: ("H512", "B"),
    245: ("H768", "A"),
    246: ("H768", "B"),
    257: ("H512", "A"),
    258: ("H512", "B"),
    259: ("H768", "A"),
    260: ("H768", "B"),
    261: ("H512", "A"),
    262: ("H512", "B"),
    263: ("H768", "A"),
    264: ("H768", "B"),
    265: ("H512", "A"),
    266: ("H512", "B"),
    267: ("H768", "A"),
    268: ("H768", "B"),
    269: ("H512", "A"),
    270: ("H512", "B"),
    273: ("H512", "B"),
}
LIMITS = {
    "H512": (512, 256, 768, 513),
    "H768": (768, 256, 1024, 769),
}
QSA_LAYERS = list(range(3, 48, 4))
POOL_SHAPES = {
    "ordinary_kv": [2, 12, 1026, 4, 2, 256],
    "qsa_compressed": [12, 1026, 1, 128],
    "qsa_raw": [12, 4, 1, 128],
    "qsa_positions": [4],
    "gdn_conv": [36, 2, 10240, 3],
    "gdn_temporal": [36, 2, 48, 128, 128],
    "ple_conv": [1, 2, 10240, 9],
    "ple_history": [2, 2],
}


def _request(rid, ids, tokens, *, stream, ignore_eos):
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


def _row_from_fixture(case, *, version, profile, block, kind, tokens, stream, ignore_eos):
    rid = f"r14-b-v{version}-{profile.lower()}-{block.lower()}-{case['request_id']}-{kind}"
    return {
        "kind": kind,
        "case_id": case["request_id"],
        "session_id": case["session_id"],
        "split": case["split"],
        "bucket": case["bucket"],
        "category": case["category"],
        "validator": case["validator"],
        "input_tokens": len(case["input_ids"]),
        "decode_tokens": tokens,
        "request": _request(
            rid, case["input_ids"], tokens, stream=stream, ignore_eos=ignore_eos
        ),
    }


def _h768_quality(rows, block):
    dev = [row for row in rows if row["split"] == "DEV"]
    holdout = []
    for bucket in ("LE128", "129_256", "257_512", "513_768"):
        selected = [
            row for row in rows
            if row["split"] == "HOLDOUT" and row["bucket"] == bucket
        ]
        holdout.extend(selected[:8] if block == "A" else selected[4:12])
    result = dev + holdout
    if len(result) != 48:
        raise RuntimeError("H768 quality split is not 48 requests")
    return result


def build_schedule(fixture, edge_fixture, *, version, profile, block):
    if ROLES.get(version) != (profile, block):
        raise ValueError("Round14 B role/version mismatch")
    source_rows = fixture.get("rows")
    if not isinstance(source_rows, list) or len(source_rows) != 64:
        raise ValueError("Round14 B frozen fixture identity mismatch")
    eligible_buckets = (
        {"LE128", "129_256", "257_512"}
        if profile == "H512"
        else {"LE128", "129_256", "257_512", "513_768"}
    )
    if profile == "H512":
        quality_cases = [row for row in source_rows if row["bucket"] in eligible_buckets]
    else:
        quality_cases = _h768_quality(source_rows, block)

    rows = [
        _row_from_fixture(
            case, version=version, profile=profile, block=block,
            kind="quality", tokens=case["normal_max_new_tokens"],
            stream=False, ignore_eos=False,
        )
        for case in quality_cases
    ]

    performance_cases = []
    for bucket in sorted(eligible_buckets):
        selected = [row for row in source_rows if row["bucket"] == bucket]
        if profile == "H512":
            chosen = (
                [row for row in selected if row["split"] == "DEV"][:4]
                if block == "A"
                else [row for row in selected if row["split"] == "HOLDOUT"][:4]
            )
        else:
            chosen = (
                [row for row in selected if row["split"] == "DEV"][:3]
                if block == "A"
                else [row for row in selected if row["split"] == "HOLDOUT"][:3]
            )
        performance_cases.extend(chosen)
    if len(performance_cases) != 12:
        raise RuntimeError("Round14 B performance distribution must contain 12 requests")
    rows.extend(
        _row_from_fixture(
            case, version=version, profile=profile, block=block,
            kind="performance", tokens=case["performance_fixed_output_tokens"],
            stream=True, ignore_eos=True,
        )
        for case in performance_cases
    )

    max_input, max_output, max_total, _threshold = LIMITS[profile]
    accept_case = "edge512" if profile == "H512" else "edge768"
    reject_case = "edge768" if profile == "H512" else "edge1024"
    accept_ids = edge_fixture["cases"][accept_case]["input_ids"]
    reject_ids = edge_fixture["cases"][reject_case]["input_ids"]
    if len(accept_ids) != max_input or len(reject_ids) <= max_input:
        raise RuntimeError("Round14 B exact boundary fixture mismatch")
    if max_input + max_output != max_total:
        raise RuntimeError("Round14 B max context identity mismatch")
    rows.append({
        "kind": "boundary_accept",
        "case_id": accept_case,
        "split": "BOUNDARY",
        "bucket": profile,
        "category": "BOUNDARY",
        "input_tokens": len(accept_ids),
        "decode_tokens": max_output,
        "request": _request(
            f"r14-b-v{version}-{profile.lower()}-{block.lower()}-boundary-accept",
            accept_ids, max_output, stream=True, ignore_eos=True,
        ),
    })
    rows.append({
        "kind": "boundary_reject",
        "case_id": reject_case,
        "split": "BOUNDARY",
        "bucket": "OUTSIDE",
        "category": "BOUNDARY",
        "input_tokens": len(reject_ids),
        "decode_tokens": 1,
        "request": _request(
            f"r14-b-v{version}-{profile.lower()}-{block.lower()}-boundary-reject",
            reject_ids, 1, stream=False, ignore_eos=False,
        ),
    })
    rows.append({
        "kind": "cancel",
        "case_id": accept_case,
        "phase": "cancel",
        "split": "CONTROL",
        "bucket": profile,
        "category": "CANCEL",
        "input_tokens": len(accept_ids),
        "decode_tokens": max_output,
        "request": _request(
            f"r14-b-v{version}-{profile.lower()}-{block.lower()}-cancel",
            accept_ids, max_output, stream=True, ignore_eos=True,
        ),
    })
    reuse_case = edge_fixture["cases"]["canary_math391"]
    rows.append({
        "kind": "reuse",
        "case_id": "canary_math391",
        "phase": "reuse",
        "split": "CONTROL",
        "bucket": "LE128",
        "category": "REUSE",
        "validator": reuse_case["validator"],
        "input_tokens": len(reuse_case["input_ids"]),
        "decode_tokens": 32,
        "request": _request(
            f"r14-b-v{version}-{profile.lower()}-{block.lower()}-reuse",
            reuse_case["input_ids"], 32, stream=False, ignore_eos=False,
        ),
    })
    if len(rows) != 64:
        raise RuntimeError(f"Round14 B schedule must contain 64 requests, got {len(rows)}")
    return rows


def _read_jsonl(path):
    path = Path(path)
    if not path.exists():
        return []
    raw = path.read_bytes()
    if raw and not raw.endswith(b"\n"):
        raise RuntimeError(f"unterminated evidence file {path.name}")
    return [json.loads(line) for line in raw.splitlines()]


def validate_qsa_state(request_id, events, *, input_tokens, decode_tokens):
    dispatch = [
        row for row in events
        if row.get("status") == "CONTEXT_REFERENCE_QSA_DISPATCH"
        and row.get("request_id") == request_id
    ]
    drains = [
        row for row in events
        if row.get("status") == "CONTEXT_REFERENCE_REQUEST_DRAINED"
        and row.get("request_id") == request_id
    ]
    if len(drains) != 1:
        raise RuntimeError("QSA request drain identity mismatch")
    epochs = {row.get("allocation_epoch") for row in dispatch + drains}
    if len(epochs) != 1 or None in epochs:
        raise RuntimeError("QSA allocation epoch is not continuous")
    prefill_sums = {}
    direct_counts = {}
    prompt_direct_steps = {}
    decode_counts = {}
    expected_decode_steps = max(0, decode_tokens - 1)
    for layer in QSA_LAYERS:
        selected = [row for row in dispatch if row.get("layer") == layer]
        prefill = [row for row in selected if row.get("strategy") == "VECTORIZED"]
        decode = [row for row in selected if row.get("strategy") == "DIRECT_BF16_ROW"]
        if any(type(row.get("query_tokens")) is not int for row in selected):
            raise RuntimeError("QSA query-token evidence is malformed")
        prefill_sums[layer] = sum(row["query_tokens"] for row in prefill)
        direct_counts[layer] = len(decode)
        prompt_direct_steps[layer] = input_tokens - prefill_sums[layer]
        decode_counts[layer] = direct_counts[layer] - prompt_direct_steps[layer]
        if (
            prefill_sums[layer] <= 0
            or prefill_sums[layer] > input_tokens
            or prompt_direct_steps[layer] not in {0, 1}
            or decode_counts[layer] != expected_decode_steps
            or any(row["query_tokens"] != 1 for row in decode)
            or len(prefill) == 0
            or prefill_sums[layer] + direct_counts[layer]
            != input_tokens + expected_decode_steps
        ):
            raise RuntimeError("QSA Prefill/Decode shape or Decode step mismatch")
    unexpected = [
        row for row in dispatch
        if row.get("layer") not in QSA_LAYERS
        or row.get("strategy") not in {"VECTORIZED", "DIRECT_BF16_ROW"}
    ]
    if unexpected:
        raise RuntimeError("QSA provider emitted an unexpected strategy/layer")
    return {
        "prefill_strategy": "VECTORIZED",
        "decode_strategy": "DIRECT_BF16_ROW",
        "layers": QSA_LAYERS,
        "prefill_tokens_per_layer": next(iter(prefill_sums.values())),
        "prompt_direct_steps_per_layer": next(iter(prompt_direct_steps.values())),
        "decode_steps_per_layer": next(iter(decode_counts.values())),
        "effective_context_forwards_per_layer": (
            next(iter(prefill_sums.values())) + next(iter(direct_counts.values()))
        ),
        "actual_completion_tokens": decode_tokens,
        "first_completion_token_from_prefill_logits": True,
        "allocation_epoch": next(iter(epochs)),
        "request_drained": True,
    }


def validate_context_pools(path):
    document = json.loads(Path(path).read_text())
    tensors = document.get("measurement", {}).get("tensors", [])
    actual = {row.get("name"): row.get("shape") for row in tensors}
    if document.get("status") != "PASS_ACTUAL_CONTEXT_POOL_GEOMETRY" or actual != POOL_SHAPES:
        raise RuntimeError("Round14 B context state-pool matrix mismatch")
    if any(row.get("device") != "npu:0" for row in tensors):
        raise RuntimeError("Round14 B state pool is not on the admitted NPU")
    return {
        "status": document["status"],
        "shapes": actual,
        "total_bytes": document["measurement"].get("total_bytes"),
        "physical_token_slots": document["footprint"].get("physical_token_slots"),
        "scope": document.get("scope"),
    }


def _finite_response(value):
    meta = value.get("meta_info")
    if not isinstance(meta, dict):
        raise RuntimeError("response metadata missing")
    logs = meta.get("output_token_logprobs")
    ids = value.get("output_ids")
    if ids is None and isinstance(logs, list):
        ids = [row[1] for row in logs]
    if (
        not isinstance(ids, list) or not ids
        or not isinstance(logs, list) or len(logs) != len(ids)
        or any(
            not isinstance(row, (list, tuple)) or not row
            or not isinstance(row[0], (int, float)) or not math.isfinite(row[0])
            for row in logs
        )
        or meta.get("completion_tokens") != len(ids)
    ):
        raise RuntimeError("response IDs/logprobs are missing or nonfinite")
    text = value.get("text")
    if not isinstance(text, str) or not text:
        raise RuntimeError("response text is empty")
    return ids, text, meta


def validate_quality(row, value):
    ids, text, meta = _finite_response(value)
    validator = row["validator"]
    kind = validator["kind"]
    if kind == "exact_text":
        passed = text.strip() == validator["expected"]
        status = "PASS_AUTOMATIC" if passed else "FAIL_AUTOMATIC"
    elif kind == "json_object":
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            parsed = None
        passed = parsed == validator["expected"]
        status = "PASS_AUTOMATIC" if passed else "FAIL_AUTOMATIC"
    elif kind == "human_review":
        passed = None
        status = validator.get("review_status", "PENDING_HUMAN")
    else:
        raise RuntimeError("unknown Round14 B validator")
    return {
        "output_ids": ids,
        "completion_tokens": len(ids),
        "output_text": text,
        "finite_logprobs": True,
        "finish_reason": meta.get("finish_reason"),
        "quality_status": status,
        "quality_pass": passed,
    }


def _boundary_event(out, request_id, expected_status):
    rows = [
        row for row in _read_jsonl(Path(out) / "round14-boundary-events.jsonl")
        if row.get("request_id") == request_id
    ]
    if (
        len(rows) != 1 or rows[0].get("status") != expected_status
        or rows[0].get("before_scheduler_dispatch") is not True
    ):
        raise RuntimeError("H-BOUNDED gate receipt mismatch")
    return rows[0]


def _provider_evidence(out, row, decode_tokens):
    from round11_t4_driver import validate_hybrid_path
    from round14_a_driver import validate_kt_ple

    out = Path(out)
    rid = row["request"]["rid"]
    return {
        "boundary": _boundary_event(out, rid, "ROUND14_H_BOUNDED_ACCEPT"),
        "hybrid": validate_hybrid_path(
            rid,
            _read_jsonl(out / "hybrid-path.jsonl"),
            _read_jsonl(out / "layerwise-events.jsonl"),
        ),
        "qsa": validate_qsa_state(
            rid,
            _read_jsonl(out / "context-reference-events.jsonl"),
            input_tokens=row["input_tokens"],
            decode_tokens=decode_tokens,
        ),
        "ple": validate_kt_ple(
            rid, _read_jsonl(out / "ple-async-events.jsonl")
        ),
    }


def _save(path, value):
    path = Path(path)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def stream_metrics(stream, started_ns, ended_ns):
    arrivals = stream["token_arrival_monotonic_ns"]
    if not arrivals or arrivals[0] < started_ns or arrivals[-1] > ended_ns:
        raise RuntimeError("Round14 B token timing is outside request wall clock")
    return {
        "ttft_seconds": (arrivals[0] - started_ns) / 1e9,
        "tpot_seconds": (
            (arrivals[-1] - arrivals[0]) / (len(arrivals) - 1) / 1e9
            if len(arrivals) > 1 else None
        ),
        "e2e_seconds": (ended_ns - started_ns) / 1e9,
    }


def run(version, profile, block, out, fixture_path, edge_fixture_path):
    from round10_t1_driver import (
        post, run_cancel, validate_cancel_receipt, validate_response,
    )
    from round11_t4_driver import consume_stream

    out = Path(out)
    result_path = out / "round14-b-raw.json"
    if result_path.exists():
        raise FileExistsError(result_path)
    fixture_path = Path(fixture_path)
    edge_fixture_path = Path(edge_fixture_path)
    fixture = json.loads(fixture_path.read_text())
    edge = json.loads(edge_fixture_path.read_text())
    schedule = build_schedule(
        fixture, edge, version=version, profile=profile, block=block
    )
    report = {
        "schema": 1,
        "status": "RUNNING",
        "stage": "ROUND14_B_REAL_DISTRIBUTION_AND_BOUNDED_HYBRID",
        "version": version,
        "profile": profile,
        "block": block,
        "fixture_sha256": hashlib.sha256(fixture_path.read_bytes()).hexdigest(),
        "edge_fixture_sha256": hashlib.sha256(edge_fixture_path.read_bytes()).hexdigest(),
        "limits": dict(zip(
            ("max_input_tokens", "max_output_tokens", "max_total_tokens", "hybrid_threshold_tokens"),
            LIMITS[profile],
        )),
        "profiler": "OFF",
        "generation_budget": 64,
        "generation_attempts": 0,
        "abort_controls": 0,
        "automatic_retries": 0,
        "state_pool_matrix": validate_context_pools(out / "context-pools.json"),
        "results": [],
        "cancel_receipts": [],
    }
    pending_cancel = None
    try:
        for ordinal, row in enumerate(schedule, 1):
            report["generation_attempts"] += 1
            rid = row["request"]["rid"]
            started = time.monotonic_ns()
            item = {
                "ordinal": ordinal,
                **{key: row[key] for key in (
                    "kind", "case_id", "session_id", "split", "bucket", "category",
                    "input_tokens", "decode_tokens",
                ) if key in row},
                "request_id": rid,
            }
            if row["kind"] == "boundary_reject":
                try:
                    with post("/generate", row["request"], 60) as response:
                        body = response.read().decode()
                    raise RuntimeError(f"outside H-BOUNDED request was accepted: {body[:200]}")
                except urllib.error.HTTPError as error:
                    body = error.read().decode(errors="replace")
                    if error.code < 400 or error.code >= 500 or "H-BOUNDED" not in body:
                        raise RuntimeError("H-BOUNDED rejection HTTP contract mismatch") from error
                    event = _boundary_event(out, rid, "ROUND14_H_BOUNDED_REJECT")
                    if any(
                        evidence.get("request_id") == rid
                        for path in ("hybrid-path.jsonl", "layerwise-events.jsonl")
                        for evidence in _read_jsonl(out / path)
                    ):
                        raise RuntimeError("rejected request reached scheduler/provider")
                    item.update(
                        expected_rejection=True, http_status=error.code,
                        response_body=body, boundary=event,
                    )
            elif row["kind"] == "cancel":
                report["abort_controls"] += 1
                pending_cancel = run_cancel(row, out, post_fn=post)
                item.update(
                    cancel_status=pending_cancel.get("status"),
                    boundary=_boundary_event(out, rid, "ROUND14_H_BOUNDED_ACCEPT"),
                )
            elif row["kind"] in {"performance", "boundary_accept"}:
                with post("/generate", row["request"], 1800) as response:
                    stream = consume_stream(response, row["decode_tokens"])
                ended = time.monotonic_ns()
                item.update(stream)
                item.update(stream_metrics(stream, started, ended))
                item["provider"] = _provider_evidence(
                    out, row, stream["completion_tokens"]
                )
            else:
                with post("/generate", row["request"], 1800) as response:
                    value = json.load(response)
                ended = time.monotonic_ns()
                if row["kind"] == "reuse":
                    validated = validate_response(row, value)
                    quality = {
                        **validated,
                        "quality_status": "PASS_AUTOMATIC",
                        "quality_pass": True,
                    }
                else:
                    quality = validate_quality(row, value)
                item.update(quality)
                item["e2e_seconds"] = (ended - started) / 1e9
                item["provider"] = _provider_evidence(
                    out, row, quality["completion_tokens"]
                )
                if row["kind"] == "reuse":
                    if pending_cancel is None:
                        raise RuntimeError("reuse lacks preceding cancellation")
                    pending_cancel["reuse"] = {
                        "request_id": rid,
                        "validator_passed": True,
                        "output_ids": quality["output_ids"],
                    }
                    pending_cancel["status"] = "PASS_HYBRID_MIDPREFILL_CANCEL_AND_REUSE"
                    validate_cancel_receipt(pending_cancel)
                    receipt_path = out / "round14-b-cancel-receipt.json"
                    _save(receipt_path, pending_cancel)
                    report["cancel_receipts"].append(receipt_path.name)
                    pending_cancel = None
            item["request_started_monotonic_ns"] = started
            item["request_ended_monotonic_ns"] = time.monotonic_ns()
            report["results"].append(item)
            _save(result_path, report)

        expected = {kind: sum(row["kind"] == kind for row in schedule) for kind in {
            "quality", "performance", "boundary_accept", "boundary_reject", "cancel", "reuse"
        }}
        actual = {kind: sum(row["kind"] == kind for row in report["results"]) for kind in expected}
        if (
            report["generation_attempts"] != 64
            or report["abort_controls"] != 1
            or len(report["results"]) != 64
            or actual != expected
            or pending_cancel is not None
            or report["cancel_receipts"] != ["round14-b-cancel-receipt.json"]
        ):
            raise RuntimeError("Round14 B schedule/budget/cancel closure is incomplete")
        report["coverage"] = {
            "scheduled": actual,
            "quality_case_ids": sorted({
                row["case_id"] for row in report["results"] if row["kind"] == "quality"
            }),
            "holdout_quality": sum(
                row.get("split") == "HOLDOUT" and row["kind"] == "quality"
                for row in report["results"]
            ),
            "automatic_quality_pass": sum(
                row.get("quality_status") == "PASS_AUTOMATIC"
                for row in report["results"]
            ),
            "automatic_quality_fail": sum(
                row.get("quality_status") == "FAIL_AUTOMATIC"
                for row in report["results"]
            ),
            "human_review_pending": sum(
                row.get("quality_status") == "PENDING_HUMAN"
                for row in report["results"]
            ),
            "expected_boundary_rejections": 1,
            "unexpected_errors": 0,
        }
        report["status"] = "PASS_ROUND14_B_DRIVER_RAW"
        _save(result_path, report)
        return report
    except BaseException as error:
        report.update(status="FAIL_ROUND14_B_DRIVER", error=repr(error))
        _save(result_path, report)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", type=int, required=True)
    parser.add_argument("--profile", choices=("H512", "H768"), required=True)
    parser.add_argument("--block", choices=("A", "B"), required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--edge-fixture", type=Path, required=True)
    args = parser.parse_args()
    result = run(
        args.version, args.profile, args.block, args.out,
        args.fixture, args.edge_fixture,
    )
    print(result["status"], args.version, args.profile, args.block, flush=True)


if __name__ == "__main__":
    main()
