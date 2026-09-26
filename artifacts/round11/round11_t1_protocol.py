"""Pure Round11 T1 fixture, schedule, response, and path contracts."""

from __future__ import annotations

import copy
import hashlib
import json
import math


MODEL_REVISION = "de4b8e4d43b917e7706784d8bb445c9af86a3540"
BASELINE_THRESHOLD = 129
FORCED_HYBRID_THRESHOLD = 1025
CHUNK_TOKENS = 32
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
    **{f"edge{length}": length for length in (128, 129, 130, 160, 192, 256, 384, 512, 768, 1024)},
    "cancel256": 256,
}

# case, phase, max_new_tokens, ignore_eos
SCHEDULES = {
    "BOUNDARY": (
        ("edge128", "boundary", 4, False),
        ("edge129", "boundary", 4, False),
        ("edge130", "boundary", 4, False),
    ),
    "HYBRID256": (
        ("canary_marker", "canary", 32, False),
        ("canary_math391", "canary", 32, False),
        ("canary_json", "canary", 32, False),
        ("canary_quote", "canary", 32, False),
        ("canary_transform", "canary", 32, False),
        ("edge129", "extended_prefill", 4, False),
        ("edge130", "extended_prefill", 4, False),
        ("edge160", "extended_prefill", 4, False),
        ("edge192", "extended_prefill", 4, False),
        ("edge256", "extended_prefill", 4, False),
        ("edge256", "repeatability", 4, False),
        ("edge192", "decode_exact", 32, True),
        ("edge256", "decode_exact", 64, True),
        ("cancel256", "cancel", 8, False),
        ("canary_math391", "reuse", 32, False),
    ),
    "HYBRID512": (
        ("edge384", "extended_prefill", 4, False),
        ("edge512", "extended_prefill", 4, False),
        ("edge192", "decode_exact", 128, True),
        ("edge512", "decode_exact", 256, True),
    ),
    "HYBRID1024": (
        ("edge768", "extended_prefill", 4, False),
        ("edge1024", "extended_prefill", 4, False),
    ),
    "HYBRID_CANCEL_REPAIR": (
        ("cancel256", "cancel", 8, False),
        ("canary_math391", "reuse", 32, False),
    ),
}

THRESHOLDS = {
    "BOUNDARY": BASELINE_THRESHOLD,
    "HYBRID256": FORCED_HYBRID_THRESHOLD,
    "HYBRID512": FORCED_HYBRID_THRESHOLD,
    "HYBRID1024": FORCED_HYBRID_THRESHOLD,
    "HYBRID_CANCEL_REPAIR": FORCED_HYBRID_THRESHOLD,
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
        raise ValueError("Round11 T1 fixture schema mismatch")
    if document.get("status") != "FROZEN_ROUND11_T1_ACTUAL_TOKENIZER_FIXTURE":
        raise ValueError("Round11 T1 fixture is not frozen")
    if document.get("model_revision") != MODEL_REVISION:
        raise ValueError("model revision identity mismatch")
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
        raise ValueError("Round11 T1 fixture case set mismatch")
    for name, fixed_length in CASE_LENGTHS.items():
        item = cases[name]
        ids = item.get("input_ids") if isinstance(item, dict) else None
        if (
            not isinstance(ids, list)
            or not ids
            or len(ids) > 1024
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
            "abort",
        }:
            raise ValueError(f"{name} validator missing")
    return document


def select_path(*, total, forward_mode, threshold):
    if type(total) is not int or total < 1:
        raise ValueError("positive request total required")
    if type(threshold) is not int or threshold < 1:
        raise ValueError("positive threshold required")
    if forward_mode != "EXTEND":
        return "HYBRID"
    return "LAYERWISE" if total >= threshold else "HYBRID"


def _request(rid, ids, max_new_tokens, ignore_eos, *, stream=False):
    return {
        "rid": rid,
        "input_ids": list(ids),
        "stream": stream,
        "sampling_params": {
            "temperature": 0,
            "top_p": 1,
            "sampling_seed": 20260920,
            "max_new_tokens": max_new_tokens,
            "ignore_eos": ignore_eos,
        },
        "return_logprob": True,
        "logprob_start_len": -1,
    }


def build_schedule(fixture, process):
    validate_fixture(fixture)
    if process not in SCHEDULES:
        raise ValueError("unknown Round11 T1 process")
    threshold = THRESHOLDS[process]
    rows = []
    for index, (case_id, phase, max_new_tokens, ignore_eos) in enumerate(
        SCHEDULES[process], start=1
    ):
        item = fixture["cases"][case_id]
        rid = f"r11-t1-{process.lower()}-{index:02d}-{case_id}-{phase}"
        rows.append(
            {
                "index": index,
                "process": process,
                "case_id": case_id,
                "phase": phase,
                "threshold": threshold,
                "expected_path": select_path(
                    total=len(item["input_ids"]),
                    forward_mode="EXTEND",
                    threshold=threshold,
                ),
                "validator": copy.deepcopy(item["validator"]),
                "request": _request(
                    rid,
                    item["input_ids"],
                    max_new_tokens,
                    ignore_eos,
                    stream=phase == "cancel",
                ),
            }
        )
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
    if row["phase"] == "decode_exact":
        expected = request["sampling_params"]["max_new_tokens"]
        if len(ids) != expected or finish != "length":
            raise RuntimeError("exact Decode workload length mismatch")
    else:
        validator = row["validator"]
        if validator["kind"] == "exact_text":
            if text != validator["expected"] or finish == "length":
                raise RuntimeError("known-answer response mismatch")
        elif validator["kind"] == "assistant_message_prefix":
            expected_ids = validator["expected_token_ids"]
            if ids[: len(expected_ids)] != expected_ids or text.strip() != validator["expected_text"]:
                raise RuntimeError("ACK response mismatch")
            if finish == "length":
                raise RuntimeError("ACK response failed to terminate at EOS")
        else:
            raise RuntimeError("abort response must use cancel receipt validator")
    return {
        "output_ids": ids,
        "completion_tokens": len(ids),
        "finite_output_logprobs": True,
        "finish_reason": meta.get("finish_reason"),
        "normal_eos": finish != "length",
    }


def expected_chunks(total):
    return [
        (start, min(CHUNK_TOKENS, total - start))
        for start in range(0, total, CHUNK_TOKENS)
    ]


def validate_path(row, hybrid_rows, layerwise_rows, streaming_rows):
    rid = row["request"]["rid"]
    total = len(row["request"]["input_ids"])
    selected = [item for item in hybrid_rows if item.get("request_id") == rid]
    layerwise = [item for item in layerwise_rows if item.get("request_id") == rid]
    streaming = [item for item in streaming_rows if item.get("request_id") == rid]
    selections = [item for item in selected if item.get("stage") == "SELECTION"]
    chunks = expected_chunks(total)
    if [(item.get("start"), item.get("batch_tokens")) for item in selections] != chunks:
        raise RuntimeError("selection evidence does not cover exact scheduler chunks")
    for item in selections:
        if (
            item.get("request_tokens") != total
            or item.get("threshold") != row["threshold"]
            or item.get("path") != row["expected_path"]
            or item.get("basis") != "REQUEST_TOTAL"
        ):
            raise RuntimeError("selection identity/path mismatch")
    providers = [item for item in selected if item.get("stage") != "SELECTION"]
    if row["expected_path"] == "HYBRID":
        if layerwise or streaming:
            raise RuntimeError("Hybrid request entered Layerwise/streaming runtime")
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
                    raise RuntimeError("Hybrid provider stages missing or duplicated")
        if len(providers) != len(chunks) * 2 * len(HYBRID_PROVIDER_STAGES):
            raise RuntimeError("Hybrid provider event count mismatch")
    else:
        if providers:
            raise RuntimeError("Layerwise request entered Hybrid providers")
        begins = [
            (item.get("start"), item.get("count"))
            for item in layerwise
            if item.get("status") == "LAYERWISE_BATCH_BEGIN"
        ]
        if begins != chunks:
            raise RuntimeError("Layerwise begin evidence does not cover scheduler chunks")
        completes = [item for item in layerwise if item.get("status") == "PREFILL_COMPLETE"]
        if len(completes) != 1:
            raise RuntimeError("Layerwise complete event missing or duplicated")
        stream_begins = [
            (item.get("start"), item.get("count"))
            for item in streaming
            if item.get("status") == "STREAMING_CHUNK_BEGIN"
        ]
        stream_drains = [
            (item.get("start"), item.get("count"))
            for item in streaming
            if item.get("status") == "STREAMING_CHUNK_DRAINED"
        ]
        if stream_begins != chunks or stream_drains != chunks:
            raise RuntimeError("streaming chunks do not match scheduler chunks")
        completes = [item for item in streaming if item.get("status") == "STREAMING_PREFILL_COMPLETE"]
        if (
            len(completes) != 1
            or completes[0].get("tokens") != total
            or completes[0].get("chunks") != len(chunks)
        ):
            raise RuntimeError("streaming complete event missing or duplicated")
    return {
        "path": row["expected_path"],
        "request_total_tokens": total,
        "scheduler_chunks": [list(pair) for pair in chunks],
        "actual_dispatch_tokens": [count for _, count in chunks],
        "hybrid_provider_event_count": len(providers),
        "layerwise_event_count": len(layerwise),
        "streaming_event_count": len(streaming),
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

    lifecycle = (
        "CANCEL_SEEN",
        "SUBMIT_FENCE_PUBLISHED",
        "REQUEST_RETIRED",
        "CANCEL_CLEANUP_COMMITTED",
    )
    statuses = receipt.get("cancel_statuses", [])
    if not set(lifecycle).issubset(set(statuses)):
        raise RuntimeError("cancel fence/cleanup/retire evidence incomplete")
    events = [
        item
        for item in receipt.get("cancel_events", [])
        if item.get("request_id") == receipt.get("request_id")
        and item.get("status") in lifecycle
    ]
    by_status = {status: [item for item in events if item.get("status") == status] for status in lifecycle}
    if any(len(by_status[status]) != 1 for status in lifecycle):
        raise RuntimeError("cancel lifecycle event missing or duplicated")
    ordered = [by_status[status][0] for status in lifecycle]
    if [item.get("event_seq") for item in ordered] != sorted(
        item.get("event_seq") for item in ordered
    ):
        raise RuntimeError("cancel lifecycle order mismatch")
    epochs = {
        (item.get("request_epoch"), item.get("allocation_epoch")) for item in ordered
    }
    if len(epochs) != 1 or None in next(iter(epochs)):
        raise RuntimeError("cancel lifecycle epoch mismatch")
    if receipt.get("late_commit") is not False or receipt.get("lease_returned") is not True:
        raise RuntimeError("cancel late commit or lease return failure")
    reuse = receipt.get("reuse")
    if not isinstance(reuse, dict) or reuse.get("validator_passed") is not True:
        raise RuntimeError("same-process reuse validation missing")
    return receipt
