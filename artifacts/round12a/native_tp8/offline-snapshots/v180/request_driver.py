"""Bounded Hybrid threshold qualification driver.

This driver compares the frozen request-total thresholds 64 and 129.  It is
deliberately host-only: it enables no profiler, adds no device synchronization,
and submits exactly seven generations in one model process.
"""

import argparse
import copy
import hashlib
import json
import math
import pathlib
import time
import urllib.request


MODES = {"BASELINE64": 64, "CANDIDATE129": 129}
HYBRID_PROVIDER_STAGES = (
    "ORIGINAL_HYBRID_APPLY",
    "CPU_SUBMIT",
    "CPU_RESULT",
    "NPU_RESIDENT_APPLY",
)
CASE_LENGTHS = {"canary27": 27, "boundary63": 63, "boundary64": 64, "boundary65": 65}
SCHEDULE = (
    ("canary27", "canary", 0),
    ("boundary64", "warmup", 0),
    ("boundary64", "measurement", 1),
    ("boundary64", "measurement", 2),
    ("boundary64", "measurement", 3),
    ("boundary63", "boundary_smoke", 1),
    ("boundary65", "boundary_smoke", 1),
)


def canonical_bytes(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()


def sha256_bytes(value):
    return hashlib.sha256(value).hexdigest()


def sha256_json(value):
    return sha256_bytes(canonical_bytes(value))


def _sha(value):
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def validate_ack_reference(document, fixture):
    expected = fixture.get("tokenizer", {}).get("tokenizer_files_sha256")
    if (not isinstance(document, dict) or document.get("verified") is not True or
            document.get("model_revision") != fixture.get("model_revision") or
            document.get("tokenizer_files_sha256") != expected or
            document.get("vocab_mapping") != {"ACK": 3897} or
            document.get("special_tokens") != {"<|im_end|>": 248046} or
            document.get("expected_first_assistant_message_ids") != [3897, 248046] or
            document.get("applies_to") != ["boundary63", "boundary64", "boundary65"]):
        raise ValueError("frozen actual-tokenizer ACK reference mismatch")
    return document


def validate_fixture(document, v71_bytes):
    """Validate frozen tokenizer evidence and the two exact inherited inputs.

    The 63/65 inputs are required to carry an actual-tokenizer receipt.  The
    launcher/build step must create that receipt by applying the frozen chat
    template; this driver rejects direct token editing and never manufactures
    token IDs at runtime.
    """
    if not isinstance(document, dict) or document.get("schema") != 1:
        raise ValueError("threshold fixture schema mismatch")
    if document.get("status") != "FROZEN_ACTUAL_TOKENIZER_FIXTURE":
        raise ValueError("threshold fixture is not frozen actual-tokenizer evidence")
    if document.get("source_v71_fixture_sha256") != sha256_bytes(v71_bytes):
        raise ValueError("source v71 fixture hash mismatch")
    try:
        v71 = json.loads(v71_bytes)
        inherited = {
            "canary27": v71["cases"][0]["request"]["input_ids"],
            "boundary64": v71["cases"][2]["request"]["input_ids"],
        }
        ack_prefix = v71["baseline_output_ids"][2][:2]
    except (KeyError, IndexError, TypeError, json.JSONDecodeError) as error:
        raise ValueError("source v71 fixture contract mismatch") from error
    tokenizer = document.get("tokenizer")
    if not isinstance(tokenizer, dict) or not _sha(tokenizer.get("chat_template_sha256")):
        raise ValueError("tokenizer/chat-template identity missing")
    files = tokenizer.get("tokenizer_files_sha256")
    if not isinstance(files, dict) or not files or any(not _sha(value) for value in files.values()):
        raise ValueError("tokenizer file hashes missing")
    if (tokenizer.get("add_generation_prompt") is not True or
            tokenizer.get("enable_thinking") is not False or
            tokenizer.get("add_special_tokens") is not False):
        raise ValueError("tokenizer chat-template options mismatch")
    cases = document.get("cases")
    if not isinstance(cases, dict) or set(cases) != set(CASE_LENGTHS):
        raise ValueError("threshold fixture cases mismatch")
    for name, expected_length in CASE_LENGTHS.items():
        case = cases[name]
        ids = case.get("input_ids") if isinstance(case, dict) else None
        if (not isinstance(ids, list) or len(ids) != expected_length or
                any(type(token) is not int or token < 0 for token in ids)):
            raise ValueError(f"{name} token length/type mismatch")
        if case.get("input_tokens") != expected_length:
            raise ValueError(f"{name} declared token count mismatch")
        if case.get("direct_token_edit") is not False:
            raise ValueError(f"{name} direct token edit is forbidden")
        if case.get("tokenization_method") != "ACTUAL_TOKENIZER_APPLY_CHAT_TEMPLATE":
            raise ValueError(f"{name} actual tokenizer method missing")
        messages, rendered = case.get("messages"), case.get("rendered_chat")
        if (not isinstance(messages, list) or not messages or
                not isinstance(rendered, str) or not rendered):
            raise ValueError(f"{name} semantic prompt/chat receipt missing")
        if (case.get("messages_sha256") != sha256_json(messages) or
                case.get("rendered_chat_sha256") != sha256_bytes(rendered.encode()) or
                case.get("input_ids_sha256") != sha256_json(ids) or
                case.get("retokenized_input_ids_sha256") != sha256_json(ids)):
            raise ValueError(f"{name} tokenizer receipt hash mismatch")
        if name in inherited and ids != inherited[name]:
            raise ValueError(f"{name} v71 exact input mismatch")
        expected_provenance = "V71_EXACT" if name in inherited else "TOKENIZER_DERIVED"
        if case.get("provenance") != expected_provenance:
            raise ValueError(f"{name} provenance mismatch")
    validator = cases["canary27"].get("validator")
    if validator != {"kind": "exact_text", "expected": "391"}:
        raise ValueError("canary27 validator mismatch")
    ack_validator = {
        "kind": "assistant_message_prefix",
        "expected_text": "ACK",
        "expected_token_ids": ack_prefix,
    }
    if len(ack_validator["expected_token_ids"]) != 2:
        raise ValueError("boundary ACK assistant-message validator mismatch")
    for name in ("boundary63", "boundary64", "boundary65"):
        embedded = cases[name].get("validator")
        if embedded is not None and embedded != ack_validator:
            raise ValueError("boundary ACK assistant-message validator mismatch")
    return document


def _request(request_id, ids, *, canary):
    return {
        "rid": request_id,
        "input_ids": list(ids),
        "sampling_params": {
            "temperature": 0,
            "top_p": 1,
            "sampling_seed": 20260920,
            "max_new_tokens": 32 if canary else 4,
            "ignore_eos": not canary,
        },
        "return_logprob": True,
        "logprob_start_len": -1,
    }


def build_schedule(fixture, mode, ack_validator=None):
    if mode not in MODES:
        raise ValueError("unknown threshold mode")
    threshold = MODES[mode]
    if ack_validator is None:
        ack_validator = fixture["cases"]["boundary64"].get("validator")
    if ack_validator is None:
        raise ValueError("separate ACK validator is required for the frozen CPU fixture")
    rows = []
    for index, (case_id, phase, repeat) in enumerate(SCHEDULE, start=1):
        ids = fixture["cases"][case_id]["input_ids"]
        suffix = case_id if phase in ("canary", "warmup", "boundary_smoke") else f"{case_id}-{repeat}"
        request_id = f"r9-plan15-threshold-{mode.lower()}-{index:02d}-{suffix}"
        rows.append({
            "index": index,
            "case_id": case_id,
            "phase": phase,
            "repeat": repeat,
            "threshold": threshold,
            "expected_path": "LAYERWISE" if len(ids) >= threshold else "HYBRID",
            "validator": copy.deepcopy(
                fixture["cases"][case_id].get("validator", ack_validator)
            ),
            "request": _request(request_id, ids, canary=phase == "canary"),
        })
    return rows


def _finish_type(value):
    return value.get("type") if isinstance(value, dict) else value


def validate_response(row, value):
    if not isinstance(value, dict) or not isinstance(value.get("meta_info"), dict):
        raise RuntimeError("response/meta_info missing")
    meta = value["meta_info"]
    request = row["request"]
    if (meta.get("id") != request["rid"] or
            meta.get("prompt_tokens") != len(request["input_ids"]) or
            meta.get("cached_tokens") != 0):
        raise RuntimeError("response identity/input/prefix mismatch")
    logs = meta.get("output_token_logprobs")
    if not isinstance(logs, list) or not logs:
        raise RuntimeError("output token logprobs missing")
    try:
        finite = all(math.isfinite(float(item[0])) for item in logs)
    except (IndexError, TypeError, ValueError) as error:
        raise RuntimeError("malformed output token logprobs") from error
    if not finite:
        raise RuntimeError("nonfinite output token logprob")
    ids = value.get("output_ids")
    if ids is None:
        ids = [item[1] for item in logs]
    if (not isinstance(ids, list) or not ids or len(ids) != len(logs) or
            meta.get("completion_tokens") != len(ids)):
        raise RuntimeError("output ids/logprobs/completion count mismatch")
    text = value.get("text")
    if not isinstance(text, str) or not text.strip():
        raise RuntimeError("empty output text")
    finish = _finish_type(meta.get("finish_reason"))
    if row["phase"] == "canary":
        if text != "391" or finish == "length":
            raise RuntimeError("frozen 391 canary mismatch")
        if ids != [18, 24, 16, 248046]:
            raise RuntimeError("frozen canary token IDs changed")
    elif finish != "length" or len(ids) != 4:
        raise RuntimeError("bounded request did not finish at exactly four tokens")
    elif ids[:2] != row["validator"]["expected_token_ids"]:
        raise RuntimeError("bounded response lacks frozen ACK assistant-message prefix")
    return {"output_ids": ids, "finite_output_logprobs": True, "finish_reason": meta["finish_reason"]}


def _chunks(total):
    return [(start, min(32, total - start)) for start in range(0, total, 32)]


def validate_path_evidence(row, hybrid_rows, layerwise_rows, streaming_rows):
    request = row["request"]
    rid, total = request["rid"], len(request["input_ids"])
    hybrid_rows = [item for item in hybrid_rows if item.get("request_id") == rid]
    layerwise_rows = [item for item in layerwise_rows if item.get("request_id") == rid]
    streaming_rows = [item for item in streaming_rows if item.get("request_id") == rid]
    selections = [item for item in hybrid_rows if item.get("stage") == "SELECTION"]
    expected_chunks = _chunks(total)
    actual_chunks = [(item.get("start"), item.get("batch_tokens")) for item in selections]
    if actual_chunks != expected_chunks:
        raise RuntimeError("selection evidence does not cover exact scheduler chunks")
    for item in selections:
        if (item.get("request_tokens") != total or item.get("threshold") != row["threshold"] or
                item.get("path") != row["expected_path"] or item.get("basis") != "REQUEST_TOTAL"):
            raise RuntimeError("selection evidence identity/path mismatch")
    provider = [item for item in hybrid_rows if item.get("stage") != "SELECTION"]
    if row["expected_path"] == "HYBRID":
        if layerwise_rows or streaming_rows:
            raise RuntimeError("hybrid request unexpectedly entered layerwise/streaming runtime")
        for start, count in expected_chunks:
            for layer in (0, 3):
                matching = [item for item in provider
                            if item.get("start") == start and item.get("batch_tokens") == count
                            and item.get("layer") == layer]
                stages = [item.get("stage") for item in matching]
                if len(stages) != len(HYBRID_PROVIDER_STAGES) or set(stages) != set(HYBRID_PROVIDER_STAGES):
                    raise RuntimeError("hybrid real provider stages missing or duplicated")
        if len(provider) != len(expected_chunks) * 2 * len(HYBRID_PROVIDER_STAGES):
            raise RuntimeError("hybrid real provider stages missing or duplicated")
    else:
        if provider:
            raise RuntimeError("layerwise request unexpectedly entered hybrid providers")
        begins = [(item.get("start"), item.get("count")) for item in layerwise_rows
                  if item.get("status") == "LAYERWISE_BATCH_BEGIN"]
        if begins != expected_chunks:
            raise RuntimeError("layerwise begin evidence does not cover exact scheduler chunks")
        complete = [item for item in layerwise_rows if item.get("status") == "PREFILL_COMPLETE"]
        if len(complete) != 1:
            raise RuntimeError("layerwise complete event missing or duplicated")
        stream_begins = [(item.get("start"), item.get("count")) for item in streaming_rows
                         if item.get("status") == "STREAMING_CHUNK_BEGIN"]
        stream_drains = [(item.get("start"), item.get("count")) for item in streaming_rows
                         if item.get("status") == "STREAMING_CHUNK_DRAINED"]
        if stream_begins != expected_chunks or stream_drains != expected_chunks:
            raise RuntimeError("three-plane streaming chunks do not match scheduler chunks")
        for item in streaming_rows:
            if item.get("status") == "STREAMING_CHUNK_DRAINED" and (
                    item.get("next_start") != item.get("start") + item.get("count") or
                    item.get("planes") != ["expert", "ple", "context"]):
                raise RuntimeError("three-plane streaming drain contract mismatch")
        stream_complete = [item for item in streaming_rows
                           if item.get("status") == "STREAMING_PREFILL_COMPLETE"]
        if (len(stream_complete) != 1 or stream_complete[0].get("tokens") != total or
                stream_complete[0].get("chunks") != len(expected_chunks)):
            raise RuntimeError("streaming complete event missing or duplicated")
    return {"path": row["expected_path"], "chunks": expected_chunks,
            "hybrid_provider_event_count": len(provider),
            "layerwise_event_count": len(layerwise_rows),
            "streaming_event_count": len(streaming_rows)}


def read_jsonl(path):
    if not path.exists():
        return []
    raw = path.read_bytes()
    if raw and not raw.endswith(b"\n"):
        raise RuntimeError(f"unterminated evidence file: {path.name}")
    try:
        return [json.loads(line) for line in raw.splitlines()]
    except json.JSONDecodeError as error:
        raise RuntimeError(f"malformed evidence file: {path.name}") from error


def post(endpoint, body, timeout=1800):
    request = urllib.request.Request(
        "http://127.0.0.1:30091" + endpoint,
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    return urllib.request.urlopen(request, timeout=timeout)


def _save(path, report):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def run(mode, out, fixture_path, v71_fixture_path, ack_reference_path, *, post_fn=post):
    out = pathlib.Path(out)
    result_path = out / "threshold-results.json"
    if result_path.exists():
        raise FileExistsError(result_path)
    v71_bytes = pathlib.Path(v71_fixture_path).read_bytes()
    fixture = json.loads(pathlib.Path(fixture_path).read_text())
    validate_fixture(fixture, v71_bytes)
    ack_reference_path = pathlib.Path(ack_reference_path)
    ack_reference = json.loads(ack_reference_path.read_text())
    validate_ack_reference(ack_reference, fixture)
    v71 = json.loads(v71_bytes)
    v71_layerwise_reference = v71["baseline_output_ids"][2]
    ack_validator = {
        "kind": "assistant_message_prefix",
        "expected_text": "ACK",
        "expected_token_ids": ack_reference["expected_first_assistant_message_ids"],
    }
    schedule = build_schedule(fixture, mode, ack_validator)
    report = {
        "schema": 1,
        "status": "RUNNING",
        "mode": mode,
        "threshold": MODES[mode],
        "streaming_threshold": MODES[mode],
        "chunk_tokens": 32,
        "prefill_and_streaming_thresholds_coupled": True,
        "selection_basis": "REQUEST_TOTAL",
        "schedule": "CANARY27_WARMUP64_MEASURE64_X3_BOUNDARY63_BOUNDARY65",
        "generation_budget": 7,
        "generation_attempts": 0,
        "abort_attempts": 0,
        "cancel_qualification": "NOT_RUN_IN_THRESHOLD_SCHEDULE",
        "retries": 0,
        "profiler": "OFF",
        "observer": "HOST_ONLY_PATH_RECEIPT",
        "performance_claim": "NARROW_DESCRIPTIVE_ONLY",
        "fixture_sha256": sha256_bytes(pathlib.Path(fixture_path).read_bytes()),
        "ack_reference_sha256": sha256_bytes(ack_reference_path.read_bytes()),
        "v71_fixture_sha256": sha256_bytes(v71_bytes),
        "attempts": [],
        "started": time.time(),
    }
    _save(result_path, report)
    try:
        for row in schedule:
            attempt = {
                "index": row["index"], "case_id": row["case_id"],
                "phase": row["phase"], "repeat": row["repeat"],
                "expected_path": row["expected_path"],
                "request": copy.deepcopy(row["request"]),
                "state": "SUBMITTING", "recorded": time.time(),
            }
            report["attempts"].append(attempt)
            report["generation_attempts"] += 1
            _save(result_path, report)
            started = time.monotonic()
            with post_fn("/generate", attempt["request"], timeout=1800) as response:
                raw = response.read().decode()
            value = json.loads(raw)
            attempt["client_e2e_seconds"] = time.monotonic() - started
            attempt["raw_response"] = raw
            attempt["parsed_response"] = value
            attempt["validation"] = validate_response(row, value)
            if row["case_id"] == "boundary64":
                attempt["v71_layerwise_reference_match"] = (
                    attempt["validation"]["output_ids"] == v71_layerwise_reference
                )
                if mode == "BASELINE64" and not attempt["v71_layerwise_reference_match"]:
                    raise RuntimeError("baseline64 output differs from frozen v71 Layerwise reference")
            attempt["path_evidence"] = validate_path_evidence(
                row, read_jsonl(out / "hybrid-path.jsonl"),
                read_jsonl(out / "layerwise-events.jsonl"),
                read_jsonl(out / "streaming-events.jsonl")
            )
            attempt["state"] = "PASS"
            _save(result_path, report)
        repeated = [attempt["validation"]["output_ids"] for attempt in report["attempts"]
                    if attempt["case_id"] == "boundary64"]
        report["same_path_64_repeat_exact"] = len(repeated) == 4 and all(
            ids == repeated[0] for ids in repeated[1:]
        )
        if not report["same_path_64_repeat_exact"]:
            raise RuntimeError("same-path 64-token output IDs were not repeat-exact")
        report["baseline64_v71_reference_exact"] = (
            all(attempt.get("v71_layerwise_reference_match") is True
                for attempt in report["attempts"] if attempt["case_id"] == "boundary64")
            if mode == "BASELINE64" else None
        )
        if (out / "request-cost.jsonl").exists():
            raise RuntimeError("request cost profiler must remain OFF")
        report["status"] = "PASS"
        return report
    except BaseException as error:
        report.update(status="FAIL", error=repr(error))
        raise
    finally:
        report["ended"] = time.time()
        _save(result_path, report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=sorted(MODES), required=True)
    parser.add_argument("--out", type=pathlib.Path, required=True)
    parser.add_argument("--fixture", type=pathlib.Path, required=True)
    parser.add_argument("--v71-fixture", type=pathlib.Path, required=True)
    parser.add_argument("--ack-reference", type=pathlib.Path, required=True)
    args = parser.parse_args()
    run(args.mode, args.out, args.fixture, args.v71_fixture, args.ack_reference)


if __name__ == "__main__":
    main()
