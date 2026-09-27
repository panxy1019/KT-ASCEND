#!/usr/bin/env python3
"""Run one four-request Round14 C fresh S0 critical-path profile block."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time


ROLES = {249: "A", 250: "B", 274: "A", 275: "B"}
QSA_LAYERS = list(range(3, 48, 4))
DEVICE_COMPONENTS = frozenset((
    "GDN", "QSA_ATTENTION", "QSA_INDEXER", "PLE_CONSUMER",
    "ROUTED_EXPERT", "NPU_RESIDENT",
))
HOST_COMPONENTS = frozenset((
    "PLE_PREPARE", "PLE_EXPOSED_WAIT", "CPU_SUBMIT", "CPU_RESULT_WAIT",
))
EXPECTED_BINDINGS = {
    "GDN": [0, 24, 46],
    "QSA_ATTENTION": [3, 23, 47],
    "QSA_INDEXER": [3, 23, 47],
    "PLE": [1],
    "EXPERT": [0, 24, 47],
}


def _ids(fixture, case_id, length):
    values = fixture.get("cases", {}).get(case_id, {}).get("input_ids")
    if (
        not isinstance(values, list) or len(values) != length
        or any(type(value) is not int for value in values)
    ):
        raise ValueError(f"Round14 C requires exact {case_id}/{length} fixture")
    return list(values)


def _request(rid, ids, tokens):
    return {
        "rid": rid,
        "input_ids": ids,
        "stream": True,
        "sampling_params": {
            "temperature": 0,
            "top_p": 1,
            "sampling_seed": 20260920,
            "max_new_tokens": tokens,
            "ignore_eos": True,
        },
        "return_logprob": True,
        "logprob_start_len": -1,
    }


def build_schedule(fixture, *, version, block):
    if ROLES.get(version) != block:
        raise ValueError("Round14 C profile role/version mismatch")
    definitions = (
        ("WARMUP", "warmup", "edge128", 128, 16),
        ("W2", "measurement", "edge128", 128, 128),
        ("W3", "measurement", "edge512", 512, 16),
        ("DRAIN", "drain", "canary_math391", 27, 4),
    )
    rows = []
    for ordinal, (workload, phase, case_id, length, decode) in enumerate(definitions, 1):
        ids = _ids(fixture, case_id, length)
        rid = f"r14-c-profile-v{version}-{block.lower()}-{ordinal:02d}-{workload.lower()}"
        rows.append({
            "ordinal": ordinal,
            "workload": workload,
            "phase": phase,
            "case_id": case_id,
            "input_tokens": length,
            "decode_tokens": decode,
            "request": _request(rid, ids, decode),
        })
    return rows


def validate_qsa(request_id, events, *, input_tokens, decode_tokens):
    rows = [
        row for row in events
        if row.get("status") == "CONTEXT_REFERENCE_QSA_DISPATCH"
        and row.get("request_id") == request_id
    ]
    prefill_sums = {}
    prompt_direct_steps = {}
    generated_decode_steps = {}
    expected_generated_decode = max(0, decode_tokens - 1)
    for layer in QSA_LAYERS:
        selected = [row for row in rows if row.get("layer") == layer]
        prefill = [row for row in selected if row.get("strategy") == "VECTORIZED"]
        direct = [row for row in selected if row.get("strategy") == "DIRECT_BF16_ROW"]
        prefill_sums[layer] = sum(row.get("query_tokens", -1) for row in prefill)
        prompt_direct_steps[layer] = input_tokens - prefill_sums[layer]
        generated_decode_steps[layer] = len(direct) - prompt_direct_steps[layer]
        if (
            not prefill
            or prefill_sums[layer] <= 0
            or prefill_sums[layer] > input_tokens
            or prompt_direct_steps[layer] not in {0, 1}
            or generated_decode_steps[layer] != expected_generated_decode
            or any(row.get("query_tokens") != 1 for row in direct)
        ):
            raise RuntimeError("Round14 C QSA path/shape/step mismatch")
    if any(
        row.get("layer") not in QSA_LAYERS
        or row.get("strategy") not in {"VECTORIZED", "DIRECT_BF16_ROW"}
        for row in rows
    ):
        raise RuntimeError("Round14 C QSA emitted an unqualified branch")
    return {
        "prefill_strategy": "VECTORIZED",
        "decode_strategy": "DIRECT_BF16_ROW",
        "layers": QSA_LAYERS,
        "prefill_tokens_per_layer": next(iter(prefill_sums.values())),
        "prompt_direct_steps_per_layer": next(iter(prompt_direct_steps.values())),
        "generated_decode_steps_per_layer": next(iter(generated_decode_steps.values())),
        "actual_completion_tokens": decode_tokens,
        "first_completion_token_from_prefill_logits": True,
        "dispatch_count": len(rows),
    }


def _read_jsonl(path):
    path = Path(path)
    if not path.exists():
        return []
    raw = path.read_bytes()
    if raw and not raw.endswith(b"\n"):
        raise RuntimeError(f"unterminated profile evidence: {path.name}")
    return [json.loads(line) for line in raw.splitlines()]


def _validate_span_batch(batch):
    if (
        batch.get("status") != "ROUND13_P0_BATCH"
        or batch.get("mode") not in {"extend", "decode"}
        or batch.get("dropped_spans") != 0
        or batch.get("observer_errors") != []
        or not isinstance(batch.get("spans"), list)
    ):
        raise RuntimeError("Round14 C profile batch metadata is incomplete")
    seen = {}
    for span in batch["spans"]:
        key = (span.get("component"), span.get("layer"))
        if key in seen:
            raise RuntimeError("Round14 C duplicate component/layer span")
        seen[key] = span
        component = span.get("component")
        expected = "COMPLETE" if component in DEVICE_COMPONENTS else "NOT_APPLICABLE"
        if (
            component not in DEVICE_COMPONENTS | HOST_COMPONENTS
            or span.get("device_status") != expected
            or span.get("observer_errors") != []
        ):
            raise RuntimeError("Round14 C profile span is unresolved")
    expected = set()
    for name, layers in (
        ("GDN", (0, 24, 46)),
        ("QSA_ATTENTION", (3, 23, 47)),
        ("QSA_INDEXER", (3, 23, 47)),
    ):
        expected.update((name, layer) for layer in layers)
    expected.update((name, 1) for name in (
        "PLE_PREPARE", "PLE_EXPOSED_WAIT", "PLE_CONSUMER",
    ))
    for layer in (0, 24, 47):
        expected.update((name, layer) for name in (
            "ROUTED_EXPERT", "CPU_SUBMIT", "CPU_RESULT_WAIT", "NPU_RESIDENT",
        ))
    if set(seen) != expected:
        raise RuntimeError("Round14 C profile sample coverage mismatch")
    return batch


def collect_batches(batches, request_ids):
    request_ids = set(request_ids)
    selected = []
    modes = {request_id: set() for request_id in request_ids}
    identities = set()
    for batch in batches:
        request_id = batch.get("request_id")
        if request_id not in request_ids:
            continue
        _validate_span_batch(batch)
        identity = (request_id, batch["mode"], batch["batch_sequence"])
        if identity in identities:
            raise RuntimeError("Round14 C duplicate profile batch")
        identities.add(identity)
        modes[request_id].add(batch["mode"])
        selected.append(batch)
    if any(modes[request_id] != {"extend", "decode"} for request_id in request_ids):
        raise RuntimeError("Round14 C profile lacks extend/decode coverage")
    return selected


def _save(path, value):
    path = Path(path)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def run(version, block, out, fixture_path):
    from round11_t4_driver import consume_stream, post, validate_hybrid_path

    out = Path(out)
    result_path = out / "round14-c-profile-raw.json"
    if result_path.exists():
        raise FileExistsError(result_path)
    fixture_path = Path(fixture_path)
    fixture = json.loads(fixture_path.read_text())
    schedule = build_schedule(fixture, version=version, block=block)
    report = {
        "schema": 1,
        "status": "RUNNING",
        "stage": "ROUND14_C_FRESH_S0_PROFILE",
        "version": version,
        "block": block,
        "parent": "ROUND13_FINAL_S0_UNIFORM_Q32_PROMOTED_QSA",
        "profile_threshold": 513,
        "default_threshold": 129,
        "profiler": "BOUNDED_OBSERVER_ON",
        "fixture_sha256": hashlib.sha256(fixture_path.read_bytes()).hexdigest(),
        "generation_budget": 4,
        "generation_attempts": 0,
        "abort_controls": 0,
        "automatic_retries": 0,
        "results": [],
        "measurement_requests": [],
        "batches": [],
    }
    try:
        for row in schedule:
            report["generation_attempts"] += 1
            started = time.monotonic_ns()
            with post("/generate", row["request"], 1800) as response:
                stream = consume_stream(response, row["decode_tokens"])
            ended = time.monotonic_ns()
            rid = row["request"]["rid"]
            qsa = validate_qsa(
                rid, _read_jsonl(out / "context-reference-events.jsonl"),
                input_tokens=row["input_tokens"], decode_tokens=row["decode_tokens"],
            )
            path = validate_hybrid_path(
                rid, _read_jsonl(out / "hybrid-path.jsonl"),
                _read_jsonl(out / "layerwise-events.jsonl"),
            )
            arrivals = stream["token_arrival_monotonic_ns"]
            item = {
                **{key: row[key] for key in (
                    "ordinal", "workload", "phase", "case_id",
                    "input_tokens", "decode_tokens",
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
                "path": path,
                "qsa": qsa,
            }
            if row["workload"] in {"W2", "W3"} and stream["output_ids"][:2] != [3897, 248046]:
                raise RuntimeError("Round14 C profile changed frozen ACK prefix")
            report["results"].append(item)
            if row["phase"] == "measurement":
                report["measurement_requests"].append(item)
            _save(result_path, report)
        bindings = json.loads((out / "round14-c-bindings.json").read_text())
        if (
            bindings.get("status") != "ROUND13_P0_ATTRIBUTION_BOUND"
            or bindings.get("bindings") != EXPECTED_BINDINGS
            or bindings.get("device_query") != "DEFERRED_QUERY_ONLY_NO_SYNCHRONIZE"
        ):
            raise RuntimeError("Round14 C observer binding mismatch")
        measurement_ids = {
            row["request_id"] for row in report["measurement_requests"]
        }
        report["batches"] = collect_batches(
            _read_jsonl(out / "round14-c-spans.jsonl"), measurement_ids
        )
        if (
            report["generation_attempts"] != 4
            or [row["workload"] for row in report["measurement_requests"]] != ["W2", "W3"]
        ):
            raise RuntimeError("Round14 C profile schedule incomplete")
        report["status"] = "PASS_ROUND14_C_PROFILE_DRIVER_RAW"
        _save(result_path, report)
        return report
    except BaseException as error:
        report.update(status="FAIL_ROUND14_C_PROFILE_DRIVER", error=repr(error))
        _save(result_path, report)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", type=int, required=True)
    parser.add_argument("--block", choices=("A", "B"), required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--fixture", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.version, args.block, args.out, args.fixture)
    print(result["status"], args.version, args.block, flush=True)


if __name__ == "__main__":
    main()
