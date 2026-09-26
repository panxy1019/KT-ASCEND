#!/usr/bin/env python3
"""Run one bounded Round11 GDN/QSA/PLE attribution process.

The first and last generations provide warmup and a later scheduler boundary
for deferred event queries.  Only the two exact-length edge128 generations are
used for hotspot accounting.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

try:  # Support both package imports in tests and direct execution in a snapshot.
    from .round11_t4_driver import (
        consume_stream,
        post,
        read_jsonl,
        validate_hybrid_path,
    )
except ImportError:  # pragma: no cover - exercised by the remote entry point.
    from round11_t4_driver import (  # type: ignore
        consume_stream,
        post,
        read_jsonl,
        validate_hybrid_path,
    )


GDN_LAYERS = (0, 24, 46)
QSA_LAYERS = (3, 23, 47)
EXPECTED_BINDINGS = {
    "GDN": list(GDN_LAYERS),
    "QSA_ATTENTION": list(QSA_LAYERS),
    "QSA_INDEXER": list(QSA_LAYERS),
    "PLE": [1],
}
EXPECTED_SAMPLES = {
    "GDN": GDN_LAYERS,
    "QSA_ATTENTION": QSA_LAYERS,
    "QSA_INDEXER": QSA_LAYERS,
    "PLE_PREPARE": (1,),
    "PLE_EXPOSED_WAIT": (1,),
    "PLE_CONSUMER": (1,),
}
DEVICE_COMPONENTS = frozenset((
    "GDN", "QSA_ATTENTION", "QSA_INDEXER", "PLE_CONSUMER",
))
HOST_COMPONENTS = frozenset(("PLE_PREPARE", "PLE_EXPOSED_WAIT"))


def _save(path, value):
    path = Path(path)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    )
    temporary.replace(path)


def _ids(fixture, case_id, expected_length):
    values = fixture.get("cases", {}).get(case_id, {}).get("input_ids")
    if (
        not isinstance(values, list)
        or len(values) != expected_length
        or any(type(value) is not int or value < 0 for value in values)
    ):
        raise ValueError(
            f"T5 requires the frozen exact {expected_length}-token {case_id} fixture"
        )
    return list(values)


def build_schedule(fixture, *, block):
    """Return the fixed four-generation bounded-attribution schedule."""
    if block not in {"A", "B"}:
        raise ValueError("invalid T5 process block")
    edge = _ids(fixture, "edge128", 128)
    canary = _ids(fixture, "canary_math391", 27)
    definitions = (
        ("canary_math391", "warmup", canary, 4),
        ("edge128", "measurement", edge, 32),
        ("edge128", "measurement", edge, 64),
        ("canary_math391", "drain", canary, 4),
    )
    rows = []
    for index, (case_id, phase, input_ids, decode_tokens) in enumerate(definitions, 1):
        request_id = (
            f"r11-t5-{block.lower()}-{index:02d}-{case_id}-{phase}-decode{decode_tokens}"
        )
        rows.append({
            "index": index,
            "case_id": case_id,
            "phase": phase,
            "decode_tokens": decode_tokens,
            "request": {
                "rid": request_id,
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


def _validate_batch_sample(batch):
    if (
        batch.get("status") != "ROUND11_B_BATCH"
        or batch.get("mode") not in {"extend", "decode"}
        or type(batch.get("batch_sequence")) is not int
        or batch["batch_sequence"] < 0
        or batch.get("dropped_spans") != 0
        or batch.get("observer_errors") != []
        or not isinstance(batch.get("spans"), list)
    ):
        raise RuntimeError("T5 sample coverage metadata is incomplete")
    seen = {}
    for span in batch["spans"]:
        key = (span.get("component"), span.get("layer"))
        if key in seen:
            raise RuntimeError("T5 sample coverage has duplicate component/layer")
        seen[key] = span
    expected = {
        (component, layer)
        for component, layers in EXPECTED_SAMPLES.items()
        for layer in layers
    }
    if set(seen) != expected:
        raise RuntimeError("T5 sample coverage does not match the frozen layer sample")
    for (component, unused_layer), span in seen.items():
        if span.get("observer_errors") != []:
            raise RuntimeError("T5 sample coverage contains an observer error")
        expected_status = "COMPLETE" if component in DEVICE_COMPONENTS else "NOT_APPLICABLE"
        if span.get("device_status") != expected_status:
            raise RuntimeError("T5 sample coverage contains incomplete device timing")
        if component not in DEVICE_COMPONENTS | HOST_COMPONENTS:
            raise RuntimeError("T5 sample coverage contains an unknown component")
    return batch


def collect_measurement_batches(batches, request_ids):
    """Select and validate every traced batch belonging to measurement requests."""
    request_ids = set(request_ids)
    if not request_ids or any(not isinstance(value, str) or not value for value in request_ids):
        raise RuntimeError("T5 measurement request identity is empty")
    selected = []
    identities = set()
    modes = {request_id: set() for request_id in request_ids}
    for batch in batches:
        request_id = batch.get("request_id")
        if request_id not in request_ids:
            continue
        _validate_batch_sample(batch)
        identity = (request_id, batch["mode"], batch["batch_sequence"])
        if identity in identities:
            raise RuntimeError("T5 sample coverage contains a duplicate batch")
        identities.add(identity)
        modes[request_id].add(batch["mode"])
        selected.append(batch)
    if any(value != {"extend", "decode"} for value in modes.values()):
        raise RuntimeError("T5 sample coverage requires extend and decode for every request")
    return sorted(
        selected,
        key=lambda row: (row["request_id"], row["batch_sequence"], row["mode"]),
    )


def _validate_measurement_quality(row, stream):
    # edge128 is the frozen ACK fixture.  Longer forced Decode continues after
    # its normal EOS, so only the frozen assistant-message prefix is asserted.
    expected = (3897, 248046)
    if tuple(stream["output_ids"][: len(expected)]) != expected:
        raise RuntimeError("T5 measurement failed the frozen ACK token prefix")
    return True


def run(block, out, fixture_path, parent_identity, *, post_fn=post):
    out = Path(out)
    result_path = out / "round11-t5-process-raw.json"
    if result_path.exists():
        raise FileExistsError(result_path)
    fixture_path = Path(fixture_path)
    fixture = json.loads(fixture_path.read_text())
    schedule = build_schedule(fixture, block=block)
    report = {
        "schema": 1,
        "status": "RUNNING",
        "block": block,
        "parent_identity": parent_identity,
        "fixture_sha256": hashlib.sha256(fixture_path.read_bytes()).hexdigest(),
        "profiler": "BOUNDED_OBSERVER_ON",
        "generation_budget": 4,
        "generation_attempts": 0,
        "automatic_retries": 0,
        "results": [],
        "requests": [],
        "batches": [],
    }
    try:
        for row in schedule:
            report["generation_attempts"] += 1
            request_id = row["request"]["rid"]
            started = time.monotonic_ns()
            with post_fn("/generate", row["request"], 1800) as response:
                stream = consume_stream(response, row["decode_tokens"])
            ended = time.monotonic_ns()
            path = validate_hybrid_path(
                request_id,
                read_jsonl(out / "hybrid-path.jsonl"),
                read_jsonl(out / "layerwise-events.jsonl"),
            )
            quality_pass = (
                _validate_measurement_quality(row, stream)
                if row["phase"] == "measurement" else True
            )
            value = {
                "request_id": request_id,
                "case_id": row["case_id"],
                "phase": row["phase"],
                "input_tokens": len(row["request"]["input_ids"]),
                "decode_tokens": row["decode_tokens"],
                "e2e_seconds": (ended - started) / 1e9,
                "quality_pass": quality_pass,
                "output_ids": stream["output_ids"],
                "finite_logprobs": stream["finite_logprobs"],
                "finish_reason": stream["finish_reason"],
                "path": "HYBRID",
                "path_evidence": path,
            }
            report["results"].append(value)
            if row["phase"] == "measurement":
                report["requests"].append({
                    key: value[key]
                    for key in (
                        "request_id", "phase", "input_tokens", "decode_tokens",
                        "e2e_seconds", "quality_pass",
                    )
                })
            _save(result_path, report)

        bindings_document = json.loads((out / "round11-b-bindings.json").read_text())
        if (
            bindings_document.get("status") != "ROUND11_B_ATTRIBUTION_BOUND"
            or bindings_document.get("bindings") != EXPECTED_BINDINGS
            or bindings_document.get("device_query")
            != "DEFERRED_QUERY_ONLY_NO_SYNCHRONIZE"
        ):
            raise RuntimeError("T5 observer binding identity mismatch")
        report["bindings"] = bindings_document["bindings"]
        measurement_ids = {row["request_id"] for row in report["requests"]}
        report["batches"] = collect_measurement_batches(
            read_jsonl(out / "round11-b-spans.jsonl"), measurement_ids
        )
        if report["generation_attempts"] != 4 or len(report["requests"]) != 2:
            raise RuntimeError("T5 process schedule incomplete")
        report["status"] = "PASS_ROUND11_T5_DRIVER_RAW"
        _save(result_path, report)
        return report
    except BaseException as error:
        report["status"] = "FAIL_ROUND11_T5_DRIVER"
        report["error"] = f"{type(error).__name__}: {error}"
        _save(result_path, report)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--block", choices=("A", "B"), required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--parent-identity", required=True)
    args = parser.parse_args()
    result = run(args.block, args.out, args.fixture, args.parent_identity)
    print(result["status"], args.block, flush=True)


if __name__ == "__main__":
    main()
