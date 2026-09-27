"""Pure Round10 T1 fixture, schedule, response, path, and cancel contracts."""

from __future__ import annotations

import copy
import hashlib
import json
import math


THRESHOLD_TOKENS = 129
CHUNK_TOKENS = 32
MODEL_REVISION = "de4b8e4d43b917e7706784d8bb445c9af86a3540"
HYBRID_PROVIDER_STAGES = (
    "ORIGINAL_HYBRID_APPLY",
    "CPU_SUBMIT",
    "CPU_RESULT",
    "NPU_RESIDENT_APPLY",
)
CASE_LENGTHS = {
    "canary_marker": None,
    "canary_math391": 27,
    "canary_json": None,
    "canary_quote": None,
    "canary_transform": None,
    "edge63": 63,
    "edge64": 64,
    "edge65": 65,
    "edge127": 127,
    "edge128": 128,
    "policy129": 129,
    "policy130": 130,
    "decode64": 64,
    "cancel128": 128,
}

SCHEDULES = {
    "A": (
        ("canary_marker", "canary", 0, 32),
        ("canary_math391", "canary", 0, 32),
        ("canary_json", "canary", 0, 32),
        ("canary_quote", "canary", 0, 32),
        ("canary_transform", "canary", 0, 32),
        ("edge64", "warmup", 0, 4),
        ("edge64", "measurement", 1, 4),
        ("edge64", "measurement", 2, 4),
        ("decode64", "decode", 32, 32),
        ("cancel128", "cancel", 0, 8),
        ("canary_math391", "reuse", 1, 32),
    ),
    "B": (
        ("edge63", "edge", 0, 4),
        ("edge64", "edge", 0, 4),
        ("edge65", "edge", 0, 4),
        ("edge127", "edge", 0, 4),
        ("edge128", "edge", 0, 4),
        ("decode64", "decode", 64, 64),
        ("edge64", "warmup", 0, 4),
        ("edge64", "measurement", 1, 4),
        ("edge64", "measurement", 2, 4),
        ("cancel128", "cancel", 0, 8),
        ("canary_math391", "reuse", 1, 32),
    ),
    "R": (
        ("canary_quote", "canary", 1, 32),
        ("canary_transform", "canary", 1, 32),
        ("edge64", "warmup", 1, 4),
        ("edge64", "measurement", 1, 4),
        ("edge64", "measurement", 2, 4),
        ("decode64", "decode", 32, 32),
        ("cancel128", "cancel", 1, 8),
        ("canary_math391", "reuse", 1, 32),
        ("edge63", "edge", 1, 4),
        ("edge64", "edge", 1, 4),
        ("edge65", "edge", 1, 4),
        ("edge127", "edge", 1, 4),
        ("edge128", "edge", 1, 4),
        ("decode64", "decode", 64, 64),
        ("edge64", "warmup", 2, 4),
        ("edge64", "measurement", 3, 4),
        ("edge64", "measurement", 4, 4),
        ("cancel128", "cancel", 2, 8),
        ("canary_math391", "reuse", 2, 32),
    ),
}


def canonical_bytes(value):
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode()


def sha256_json(value):
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def _is_sha(value):
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def validate_fixture(document):
    if not isinstance(document, dict) or document.get("schema") != 1:
        raise ValueError("T1 fixture schema mismatch")
    if document.get("status") != "FROZEN_ROUND10_T1_ACTUAL_TOKENIZER_FIXTURE":
        raise ValueError("T1 fixture is not frozen")
    revision = document.get("model_revision")
    if (
        not isinstance(revision, str)
        or len(revision) != 40
        or any(character not in "0123456789abcdef" for character in revision)
    ):
        raise ValueError("model revision identity missing")
    if document.get("threshold_tokens") != THRESHOLD_TOKENS:
        raise ValueError("threshold identity mismatch")
    if document.get("chunk_tokens") != CHUNK_TOKENS:
        raise ValueError("chunk identity mismatch")
    tokenizer = document.get("tokenizer")
    if not isinstance(tokenizer, dict) or not _is_sha(tokenizer.get("chat_template_sha256")):
        raise ValueError("tokenizer identity missing")
    files = tokenizer.get("tokenizer_files_sha256")
    if (
        not isinstance(files, dict)
        or set(files) != {"tokenizer.json", "tokenizer_config.json"}
        or any(not _is_sha(value) for value in files.values())
    ):
        raise ValueError("tokenizer file identity mismatch")
    if (
        tokenizer.get("add_generation_prompt") is not True
        or tokenizer.get("enable_thinking") is not False
        or tokenizer.get("add_special_tokens") is not False
    ):
        raise ValueError("tokenizer options mismatch")
    cases = document.get("cases")
    if not isinstance(cases, dict) or set(cases) != set(CASE_LENGTHS):
        raise ValueError("T1 fixture case set mismatch")
    for name, fixed_length in CASE_LENGTHS.items():
        item = cases[name]
        ids = item.get("input_ids") if isinstance(item, dict) else None
        if (
            not isinstance(ids, list)
            or not ids
            or len(ids) > 130
            or any(type(token) is not int or token < 0 for token in ids)
        ):
            raise ValueError(f"{name} input IDs invalid")
        if fixed_length is not None and len(ids) != fixed_length:
            raise ValueError(f"{name} exact token length mismatch")
        if item.get("input_tokens") != len(ids):
            raise ValueError(f"{name} declared token length mismatch")
        if item.get("direct_token_edit") is not False:
            raise ValueError(f"{name} direct token edit forbidden")
        if item.get("tokenization_method") != "ACTUAL_TOKENIZER_APPLY_CHAT_TEMPLATE":
            raise ValueError(f"{name} tokenizer method mismatch")
        messages, rendered = item.get("messages"), item.get("rendered_chat")
        if not isinstance(messages, list) or not messages or not isinstance(rendered, str) or not rendered:
            raise ValueError(f"{name} semantic receipt missing")
        if (
            item.get("input_ids_sha256") != sha256_json(ids)
            or item.get("retokenized_input_ids_sha256") != sha256_json(ids)
            or item.get("messages_sha256") != sha256_json(messages)
            or item.get("rendered_chat_sha256")
            != hashlib.sha256(rendered.encode()).hexdigest()
        ):
            raise ValueError(f"{name} receipt hash mismatch")
        validator = item.get("validator")
        if not isinstance(validator, dict) or validator.get("kind") not in {
            "exact_text",
            "assistant_message_prefix",
            "multi_step",
            "abort",
        }:
            raise ValueError(f"{name} validator missing")
    return document


def select_path(*, total, forward_mode, threshold=THRESHOLD_TOKENS):
    if type(total) is not int or total < 1:
        raise ValueError("positive request total required")
    if type(threshold) is not int or threshold < 1:
        raise ValueError("positive threshold required")
    if forward_mode != "EXTEND":
        return "HYBRID"
    return "LAYERWISE" if total >= threshold else "HYBRID"


def _request(rid, ids, max_new_tokens, *, stream=False):
    return {
        "rid": rid,
        "input_ids": list(ids),
        "stream": stream,
        "sampling_params": {
            "temperature": 0,
            "top_p": 1,
            "sampling_seed": 20260920,
            "max_new_tokens": max_new_tokens,
            "ignore_eos": False,
        },
        "return_logprob": True,
        "logprob_start_len": -1,
    }


def build_schedule(fixture, process):
    validate_fixture(fixture)
    if process not in SCHEDULES:
        raise ValueError("unknown T1 process")
    rows = []
    for index, (case_id, phase, repeat, max_new_tokens) in enumerate(
        SCHEDULES[process], start=1
    ):
        item = fixture["cases"][case_id]
        rid = f"r10-t1-{process.lower()}-{index:02d}-{case_id}-{phase}"
        rows.append(
            {
                "index": index,
                "process": process,
                "case_id": case_id,
                "phase": phase,
                "repeat": repeat,
                "expected_path": select_path(total=len(item["input_ids"]), forward_mode="EXTEND"),
                "validator": copy.deepcopy(item["validator"]),
                "request": _request(
                    rid,
                    item["input_ids"],
                    max_new_tokens,
                    stream=phase == "cancel",
                ),
            }
        )
    expected = 19 if process == "R" else 11
    if len(rows) != expected:
        raise AssertionError("T1 process generation schedule changed")
    return rows


def _finish_type(value):
    return value.get("type") if isinstance(value, dict) else value


def validate_response(row, value):
    if not isinstance(value, dict) or not isinstance(value.get("meta_info"), dict):
        raise RuntimeError("response metadata missing")
    meta = value["meta_info"]
    request = row["request"]
    if (
        meta.get("id") != request["rid"]
        or meta.get("prompt_tokens") != len(request["input_ids"])
        or meta.get("cached_tokens") != 0
    ):
        raise RuntimeError("response identity/input/prefix mismatch")
    logs = meta.get("output_token_logprobs")
    if not isinstance(logs, list) or not logs:
        raise RuntimeError("output token logprobs missing")
    try:
        if not all(math.isfinite(float(item[0])) for item in logs):
            raise RuntimeError("nonfinite output token logprob")
    except (IndexError, TypeError, ValueError) as error:
        raise RuntimeError("malformed output token logprob") from error
    ids = value.get("output_ids")
    if ids is None:
        ids = [item[1] for item in logs]
    if (
        not isinstance(ids, list)
        or not ids
        or len(ids) != len(logs)
        or meta.get("completion_tokens") != len(ids)
    ):
        raise RuntimeError("output IDs/logprobs/count mismatch")
    text = value.get("text")
    if not isinstance(text, str) or not text:
        raise RuntimeError("empty output text")
    finish = _finish_type(meta.get("finish_reason"))
    validator = row["validator"]
    kind = validator["kind"]
    if kind == "exact_text":
        if text != validator["expected"] or finish == "length":
            raise RuntimeError("known-answer response mismatch")
    elif kind == "assistant_message_prefix":
        expected_ids = validator.get("expected_token_ids")
        if ids[: len(expected_ids)] != expected_ids or text.strip() != validator["expected_text"]:
            raise RuntimeError("ACK response mismatch")
        if finish == "length":
            raise RuntimeError("ACK response failed to terminate at EOS")
    elif kind == "multi_step":
        minimum = validator.get("minimum_tokens")
        if type(minimum) is not int or len(ids) < minimum:
            raise RuntimeError("multi-step response too short")
    else:
        raise RuntimeError("abort response must use cancel receipt validator")
    return {
        "output_ids": ids,
        "completion_tokens": len(ids),
        "finite_output_logprobs": True,
        "finish_reason": meta.get("finish_reason"),
    }


def expected_chunks(total):
    return [(start, min(CHUNK_TOKENS, total - start)) for start in range(0, total, CHUNK_TOKENS)]


def validate_hybrid_path(row, hybrid_rows, layerwise_rows, streaming_rows):
    rid = row["request"]["rid"]
    total = len(row["request"]["input_ids"])
    selected = [item for item in hybrid_rows if item.get("request_id") == rid]
    layerwise = [item for item in layerwise_rows if item.get("request_id") == rid]
    streaming = [item for item in streaming_rows if item.get("request_id") == rid]
    if layerwise or streaming:
        raise RuntimeError("Hybrid request entered Layerwise/streaming runtime")
    selections = [item for item in selected if item.get("stage") == "SELECTION"]
    chunks = expected_chunks(total)
    if [(item.get("start"), item.get("batch_tokens")) for item in selections] != chunks:
        raise RuntimeError("Hybrid selection chunks mismatch")
    for item in selections:
        if (
            item.get("request_tokens") != total
            or item.get("threshold") != THRESHOLD_TOKENS
            or item.get("path") != "HYBRID"
            or item.get("basis") != "REQUEST_TOTAL"
        ):
            raise RuntimeError("Hybrid selection identity mismatch")
    providers = [item for item in selected if item.get("stage") != "SELECTION"]
    for start, count in chunks:
        for layer in (0, 3):
            stages = [
                item.get("stage")
                for item in providers
                if item.get("start") == start
                and item.get("batch_tokens") == count
                and item.get("layer") == layer
            ]
            if len(stages) != len(HYBRID_PROVIDER_STAGES) or set(stages) != set(HYBRID_PROVIDER_STAGES):
                raise RuntimeError("Hybrid real provider stages missing or duplicated")
    if len(providers) != len(chunks) * 2 * len(HYBRID_PROVIDER_STAGES):
        raise RuntimeError("Hybrid provider event count mismatch")
    return {
        "path": "HYBRID",
        "chunks": [list(pair) for pair in chunks],
        "hybrid_provider_event_count": len(providers),
    }


def validate_cancel_receipt(receipt):
    if receipt.get("status") != "PASS_HYBRID_MIDPREFILL_CANCEL_AND_REUSE":
        raise RuntimeError("cancel receipt status mismatch")
    if receipt.get("trigger_start") != 32 or receipt.get("abort_http_status") != 200:
        raise RuntimeError("cancel trigger/HTTP acknowledgement mismatch")
    if receipt.get("generation_thread_alive") is not False:
        raise RuntimeError("cancel generation did not terminate")
    if receipt.get("terminal_finish_reason") != "abort":
        raise RuntimeError("terminal abort response missing")
    starts = [
        item.get("start")
        for item in receipt.get("hybrid_events", [])
        if item.get("request_id") == receipt.get("request_id")
        and item.get("stage") == "SELECTION"
    ]
    if 32 not in starts:
        raise RuntimeError("cancel trigger chunk missing")
    if any(type(start) is int and start > 32 for start in starts):
        raise RuntimeError("post-cancel chunk observed")
    statuses = receipt.get("cancel_statuses", [])
    if "CANCEL_SEEN" not in statuses or "CANCEL_CLEANUP_COMMITTED" not in statuses:
        raise RuntimeError("cancel cleanup lifecycle incomplete")
    if receipt.get("late_commit") is not False:
        raise RuntimeError("late commit after cancel")
    if receipt.get("lease_returned") is not True:
        raise RuntimeError("cancel lease was not returned")
    reuse = receipt.get("reuse")
    if not isinstance(reuse, dict) or reuse.get("validator_passed") is not True:
        raise RuntimeError("post-cancel reuse validation missing")
    return True
