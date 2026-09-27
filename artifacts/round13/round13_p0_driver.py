#!/usr/bin/env python3
"""Run one bounded Round13 P0 S0 attribution and DEV calibration block."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

try:
    from .round11_t4_driver import consume_stream, post, read_jsonl, validate_hybrid_path
except ImportError:  # remote snapshot entry point
    try:
        from round11_t4_driver import consume_stream, post, read_jsonl, validate_hybrid_path
    except ImportError:  # local unit tests exercise the pure schedule helpers
        consume_stream = post = read_jsonl = validate_hybrid_path = None


QSA_LAYERS = list(range(3, 48, 4))
EXPECTED_BINDINGS = {
    "GDN": [0, 24, 46],
    "QSA_ATTENTION": [3, 23, 47],
    "QSA_INDEXER": [3, 23, 47],
    "PLE": [1],
    "EXPERT": [0, 24, 47],
}
DEVICE_COMPONENTS = frozenset((
    "GDN", "QSA_ATTENTION", "QSA_INDEXER", "PLE_CONSUMER",
    "ROUTED_EXPERT", "NPU_RESIDENT",
))
HOST_COMPONENTS = frozenset((
    "PLE_PREPARE", "PLE_EXPOSED_WAIT", "CPU_SUBMIT", "CPU_RESULT_WAIT",
))


def _save(path, value):
    path = Path(path)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                    allow_nan=False) + "\n")
    temporary.replace(path)


def _ids(fixture, case_id, expected_length):
    values = fixture.get("cases", {}).get(case_id, {}).get("input_ids")
    if (not isinstance(values, list) or len(values) != expected_length or
            any(type(value) is not int or value < 0 for value in values)):
        raise ValueError(f"P0 requires exact {expected_length}-token {case_id}")
    return list(values)


def build_schedule(fixture, dev_fixture, *, block):
    if block not in {"A", "B"}:
        raise ValueError("invalid P0 block")
    dev = dev_fixture.get("prompts")
    if (dev_fixture.get("status") != "FROZEN_ROUND13_DEV_TOKEN_IDS" or
            not isinstance(dev, list) or len(dev) != 2):
        raise ValueError("invalid Round13 DEV fixture")
    edge128 = _ids(fixture, "edge128", 128)
    edge512 = _ids(fixture, "edge512", 512)
    canary = _ids(fixture, "canary_math391", 27)
    definitions = [
        (dev[0]["id"], "dev", list(dev[0]["input_ids"]), 4),
        (dev[1]["id"], "dev", list(dev[1]["input_ids"]), 4),
        ("edge128", "warmup", edge128, 128),
        ("edge128", "measurement", edge128, 128),
        ("edge512", "warmup", edge512, 16),
        ("edge512", "measurement", edge512, 16),
        ("canary_math391", "drain", canary, 4),
    ]
    rows = []
    for index, (case_id, phase, input_ids, decode_tokens) in enumerate(definitions, 1):
        prefix = "r13-dev" if phase == "dev" else f"r13-p0-{block.lower()}"
        rid = f"{prefix}-{index:02d}-{case_id}-{phase}-decode{decode_tokens}"
        rows.append({
            "index": index,
            "case_id": case_id,
            "phase": phase,
            "input_tokens": len(input_ids),
            "decode_tokens": decode_tokens,
            "request": {
                "rid": rid,
                "input_ids": input_ids,
                "stream": True,
                "sampling_params": {
                    "temperature": 0,
                    "top_p": 1,
                    "sampling_seed": 20260920,
                    "max_new_tokens": decode_tokens,
                    "ignore_eos": True,
                },
                "return_logprob": True,
                "logprob_start_len": -1,
            },
        })
    return rows


def timing_metrics(started_ns, ended_ns, arrivals):
    if not arrivals or arrivals[0] < started_ns or ended_ns < arrivals[-1]:
        raise ValueError("stream timing order is invalid")
    return {
        "ttft_seconds": (arrivals[0] - started_ns) / 1e9,
        "e2e_seconds": (ended_ns - started_ns) / 1e9,
        "tpot_seconds": ((arrivals[-1] - arrivals[0]) / (len(arrivals) - 1) / 1e9
                         if len(arrivals) > 1 else None),
    }


def _validate_qsa(request_id, events):
    rows = [row for row in events
            if row.get("status") == "CONTEXT_REFERENCE_QSA_DISPATCH"
            and row.get("request_id") == request_id]
    vector = sorted({row.get("layer") for row in rows
                     if row.get("strategy") == "VECTORIZED"})
    decode = sorted({row.get("layer") for row in rows
                     if row.get("strategy") == "ROW"})
    if vector != QSA_LAYERS or decode != QSA_LAYERS:
        raise RuntimeError("P0 did not execute promoted Prefill and ROW Decode QSA")
    return {"vectorized_layers": vector, "row_layers": decode,
            "dispatch_count": len(rows)}


def _validate_span_batch(batch):
    if (batch.get("status") != "ROUND13_P0_BATCH" or
            batch.get("mode") not in {"extend", "decode"} or
            batch.get("dropped_spans") != 0 or
            batch.get("observer_errors") != [] or
            not isinstance(batch.get("spans"), list)):
        raise RuntimeError("P0 attribution batch metadata is incomplete")
    seen = {}
    for span in batch["spans"]:
        key = (span.get("component"), span.get("layer"))
        if key in seen:
            raise RuntimeError("P0 attribution has duplicate component/layer")
        seen[key] = span
        component = span.get("component")
        expected = "COMPLETE" if component in DEVICE_COMPONENTS else "NOT_APPLICABLE"
        if component not in DEVICE_COMPONENTS | HOST_COMPONENTS:
            raise RuntimeError("P0 attribution contains unknown component")
        if span.get("device_status") != expected or span.get("observer_errors") != []:
            raise RuntimeError("P0 attribution contains unresolved timing")
    expected = set()
    for name, layers in (("GDN", (0, 24, 46)),
                         ("QSA_ATTENTION", (3, 23, 47)),
                         ("QSA_INDEXER", (3, 23, 47))):
        expected.update((name, layer) for layer in layers)
    expected.update((name, 1) for name in
                    ("PLE_PREPARE", "PLE_EXPOSED_WAIT", "PLE_CONSUMER"))
    for layer in (0, 24, 47):
        expected.update((name, layer) for name in
                        ("ROUTED_EXPERT", "CPU_SUBMIT", "CPU_RESULT_WAIT",
                         "NPU_RESIDENT"))
    if set(seen) != expected:
        raise RuntimeError("P0 attribution sample coverage mismatch")
    return batch


def collect_measurement_batches(batches, request_ids):
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
            raise RuntimeError("duplicate P0 attribution batch")
        identities.add(identity)
        modes[request_id].add(batch["mode"])
        selected.append(batch)
    if any(value != {"extend", "decode"} for value in modes.values()):
        raise RuntimeError("P0 measurement lacks extend/decode coverage")
    return sorted(selected, key=lambda row: (
        row["request_id"], row["batch_sequence"], row["mode"]))


def _validate_routes(document, dev_request_ids):
    if (document.get("status") != "ROUND13_FRESH_DEV_ROUTE_COUNTS" or
            document.get("copy_scope") != "DEV_ONLY_TOPK_IDS_D2H" or
            set(document.get("requests", [])) != set(dev_request_ids)):
        raise RuntimeError("P0 DEV route identity mismatch")
    counts = document.get("counts")
    if (not isinstance(counts, dict) or set(counts) != {"extend", "decode"} or
            any(len(counts[mode]) != 48 for mode in counts) or
            any(len(layer) != 512 for mode in counts for layer in counts[mode])):
        raise RuntimeError("P0 DEV route count shape mismatch")
    if any(document.get("assignment_totals", {}).get(mode, 0) <= 0
           for mode in ("extend", "decode")):
        raise RuntimeError("P0 DEV route mode is empty")
    return document


def run(block, out, fixture_path, dev_fixture_path, parent_identity, *, post_fn=None):
    if post_fn is None:
        if post is None:
            raise RuntimeError("Round13 P0 HTTP driver dependency is unavailable")
        post_fn = post
    out = Path(out)
    result_path = out / "round13-p0-raw.json"
    if result_path.exists():
        raise FileExistsError(result_path)
    fixture_path, dev_fixture_path = Path(fixture_path), Path(dev_fixture_path)
    fixture = json.loads(fixture_path.read_text())
    dev_fixture = json.loads(dev_fixture_path.read_text())
    schedule = build_schedule(fixture, dev_fixture, block=block)
    report = {
        "schema": 1,
        "status": "RUNNING",
        "block": block,
        "parent_identity": parent_identity,
        "profile_threshold": 513,
        "default_threshold": 129,
        "profiler": "BOUNDED_OBSERVER_ON",
        "fixture_sha256": hashlib.sha256(fixture_path.read_bytes()).hexdigest(),
        "dev_fixture_sha256": hashlib.sha256(dev_fixture_path.read_bytes()).hexdigest(),
        "generation_budget": 7,
        "generation_attempts": 0,
        "abort_controls": 0,
        "automatic_retries": 0,
        "results": [],
        "requests": [],
        "batches": [],
    }
    try:
        for row in schedule:
            report["generation_attempts"] += 1
            started = time.monotonic_ns()
            with post_fn("/generate", row["request"], 1800) as response:
                stream = consume_stream(response, row["decode_tokens"])
            ended = time.monotonic_ns()
            if stream.get("finite_logprobs") is not True:
                raise RuntimeError("P0 request produced nonfinite logprobs")
            request_id = row["request"]["rid"]
            path = validate_hybrid_path(
                request_id, read_jsonl(out / "hybrid-path.jsonl"),
                read_jsonl(out / "layerwise-events.jsonl"))
            qsa = _validate_qsa(
                request_id, read_jsonl(out / "context-reference-events.jsonl"))
            metrics = timing_metrics(started, ended,
                                     stream["token_arrival_monotonic_ns"])
            if row["case_id"] == "edge128" and stream["output_ids"][:2] != [3897, 248046]:
                raise RuntimeError("P0 W2 changed frozen ACK prefix")
            value = {
                **{key: row[key] for key in
                   ("index", "case_id", "phase", "input_tokens", "decode_tokens")},
                "request_id": request_id,
                **metrics,
                "completion_tokens": stream["completion_tokens"],
                "output_ids": stream["output_ids"],
                "finite_logprobs": stream["finite_logprobs"],
                "finish_reason": stream["finish_reason"],
                "path": "HYBRID",
                "path_evidence": path,
                "qsa": qsa,
            }
            report["results"].append(value)
            if row["phase"] == "measurement":
                report["requests"].append(value)
            _save(result_path, report)

        bindings = json.loads((out / "round13-bindings.json").read_text())
        if (bindings.get("status") != "ROUND13_P0_ATTRIBUTION_BOUND" or
                bindings.get("bindings") != EXPECTED_BINDINGS or
                bindings.get("device_query") != "DEFERRED_QUERY_ONLY_NO_SYNCHRONIZE"):
            raise RuntimeError("P0 observer binding mismatch")
        measurement_ids = {row["request_id"] for row in report["requests"]}
        report["batches"] = collect_measurement_batches(
            read_jsonl(out / "round13-spans.jsonl"), measurement_ids)
        dev_ids = [row["request"]["rid"] for row in schedule if row["phase"] == "dev"]
        report["dev_routes"] = _validate_routes(
            json.loads((out / "round13-dev-route-counts.json").read_text()), dev_ids)
        if report["generation_attempts"] != 7 or len(report["requests"]) != 2:
            raise RuntimeError("P0 process schedule incomplete")
        report["status"] = "PASS_ROUND13_P0_DRIVER_RAW"
        _save(result_path, report)
        return report
    except BaseException as error:
        report["status"] = "FAIL_ROUND13_P0_DRIVER"
        report["error"] = f"{type(error).__name__}: {error}"
        _save(result_path, report)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--block", choices=("A", "B"), required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--dev-fixture", type=Path, required=True)
    parser.add_argument("--parent-identity", required=True)
    args = parser.parse_args()
    result = run(args.block, args.out, args.fixture, args.dev_fixture,
                 args.parent_identity)
    print(result["status"], args.block, flush=True)


if __name__ == "__main__":
    main()
