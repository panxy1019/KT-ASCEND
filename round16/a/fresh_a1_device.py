"""Strict single-process HTTP driver for one fresh Round16 A1 block.

The module is import-safe and runtime-neutral.  Network access occurs only
when :func:`drive_schedule_item` is called by the explicitly enabled device
controller.  Each call issues exactly one request and contains no retry loop.
"""

from __future__ import annotations

import importlib.util
import hashlib
import json
from pathlib import Path
from typing import Any, Callable, Mapping

from .fresh_a1_profile import BLOCK_IDS, manifest_payload


FIXTURE_SHA256 = "21e4f82531a58881f2fa251fda50ef3f9909c589336164b77fe732a77abdca95"
INPUT_IDS_SHA256 = "4d2584778859d8b34fa388d6fb09925c1da1a0197945de1298f09d199dce489d"
MESSAGES_SHA256 = "e5a27eaad7efa96e5ff9020ca99f901cf968238ac7ae7b662c7369923c5e6615"
SOURCE_IDENTITY_KEYS = (
    "model_revision",
    "fixture_sha256",
    "observer_contract",
    "provider_state",
    "schedule_owner",
)


class DeviceDriverError(RuntimeError):
    """The frozen request or its response did not satisfy the A1 contract."""


def _exact_schedule_item(item: Mapping[str, Any]) -> dict[str, Any]:
    ordinal = item.get("ordinal")
    schedule = manifest_payload()["schedule"]
    if type(ordinal) is not int or not 1 <= ordinal <= len(schedule):
        raise DeviceDriverError("schedule ordinal is invalid")
    expected = schedule[ordinal - 1]
    if dict(item) != expected:
        raise DeviceDriverError("schedule item differs from the frozen six-request queue")
    return expected


def build_request(
    schedule_item: Mapping[str, Any],
    *,
    request_id: str,
    fixture: Mapping[str, Any],
) -> dict[str, Any]:
    item = _exact_schedule_item(schedule_item)
    case_id = "canary_math391" if item["role"] in {"SHORT_DRAIN", "FIXED_CANARY"} else "edge128"
    case = fixture.get("cases", {}).get(case_id)
    ids = case.get("input_ids") if isinstance(case, Mapping) else None
    if (
        not isinstance(ids, list)
        or len(ids) != item["input_tokens"]
        or any(type(token) is not int or token < 0 for token in ids)
    ):
        raise DeviceDriverError(f"fixture does not contain exact {case_id} input IDs")
    if item["role"] in {"SHORT_DRAIN", "FIXED_CANARY"}:
        if (
            case.get("input_ids_sha256") != INPUT_IDS_SHA256
            or case.get("messages_sha256") != MESSAGES_SHA256
        ):
            raise DeviceDriverError("canary fixture identity differs")
    tokens = item.get("max_new_tokens", item.get("output_tokens"))
    ignore_eos = item.get("ignore_eos", item["role"] != "FIXED_CANARY")
    return {
        "rid": request_id,
        "input_ids": list(ids),
        "stream": item["role"] != "FIXED_CANARY",
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


def _source_identity(fixture: Mapping[str, Any]) -> dict[str, Any]:
    revision = fixture.get("model_revision")
    if not isinstance(revision, str) or len(revision) != 40:
        raise DeviceDriverError("fixture model revision identity is missing")
    return {
        "model_revision": revision,
        "fixture_sha256": FIXTURE_SHA256,
        "observer_contract": "round16-a1-observer-v001",
        "provider_state": "OFF",
        "schedule_owner": "round16-contract-66",
    }


def observed_request(
    item: Mapping[str, Any], *, block_id: str, request_id: str
) -> dict[str, Any]:
    return {
        "block_id": block_id,
        "request_id": request_id,
        "role": item["role"],
        "input_tokens": item["input_tokens"],
        "max_new_tokens": item.get("max_new_tokens", item.get("output_tokens")),
        "ignore_eos": item.get("ignore_eos", False),
        "fixture_sha256": item.get("fixture_sha256"),
        "input_ids_sha256": item.get("input_ids_sha256"),
        "messages_sha256": item.get("messages_sha256"),
    }


def drive_schedule_item(
    schedule_item: Mapping[str, Any],
    *,
    request_id: str,
    fixture_path: str | Path,
    block_id: str,
    post_fn: Callable[..., Any],
    consume_stream_fn: Callable[..., Mapping[str, Any]],
    validate_response_fn: Callable[[Mapping[str, Any], Mapping[str, Any]], Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Issue one HTTP generation attempt.  There is intentionally no retry."""
    if block_id not in BLOCK_IDS.values():
        raise DeviceDriverError("unknown immutable block identity")
    item = _exact_schedule_item(schedule_item)
    try:
        fixture = json.loads(Path(fixture_path).read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise DeviceDriverError(f"cannot read frozen fixture: {exc}") from exc
    request = build_request(item, request_id=request_id, fixture=fixture)
    tokens = request["sampling_params"]["max_new_tokens"]
    with post_fn("/generate", request, 1800) as response:
        if request["stream"]:
            result = dict(consume_stream_fn(response, tokens))
            if (
                result.get("completion_tokens") != tokens
                or result.get("finite_logprobs") is not True
                or result.get("finish_reason") != "length"
            ):
                raise DeviceDriverError("stream response identity or token budget mismatch")
        else:
            try:
                value = json.loads(response.read())
            except (UnicodeError, json.JSONDecodeError) as exc:
                raise DeviceDriverError("canary response is not valid JSON") from exc
            if validate_response_fn is None:
                raise DeviceDriverError("fixed canary requires the frozen response validator")
            result = dict(
                validate_response_fn(
                    {"request": request, "validator": item["validator"]}, value
                )
            )
            if result.get("finite_output_logprobs") is not True:
                raise DeviceDriverError("fixed canary logprob validation is incomplete")
    return {
        "status": "COMPLETE",
        "observed_request": observed_request(item, block_id=block_id, request_id=request_id),
        "source_identity": _source_identity(fixture),
        "response": result,
        "automatic_retries": 0,
    }


def _load(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise DeviceDriverError(f"cannot load frozen driver module {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def driver(
    schedule_item: Mapping[str, Any],
    *,
    request_id: str,
    output_dir: Path,
    snapshot_root: Path,
    environment: Mapping[str, str],
) -> dict[str, Any]:
    """Adapter used by ``execute_controller`` inside the model container."""
    if environment.get("ROUND16_A1_PROFILE_ENABLE") != "1":
        raise DeviceDriverError("device driver requires explicit profile enablement")
    block_id = environment.get("ROUND16_A1_BLOCK_ID")
    if block_id not in BLOCK_IDS.values():
        raise DeviceDriverError("device driver block identity is missing")
    fixture_path = Path(snapshot_root) / "round10-t1-fixture.json"
    if hashlib.sha256(fixture_path.read_bytes()).hexdigest() != FIXTURE_SHA256:
        raise DeviceDriverError("frozen fixture byte SHA differs before request")
    base = _load(Path(snapshot_root) / "round11_t4_driver.py", "_r16_round11_t4")
    protocol = _load(Path(snapshot_root) / "round10_t1_protocol.py", "_r16_round10_t1")
    return drive_schedule_item(
        schedule_item,
        request_id=request_id,
        fixture_path=fixture_path,
        block_id=block_id,
        post_fn=base.post,
        consume_stream_fn=base.consume_stream,
        validate_response_fn=protocol.validate_response,
    )


__all__ = (
    "DeviceDriverError",
    "SOURCE_IDENTITY_KEYS",
    "build_request",
    "drive_schedule_item",
    "driver",
    "observed_request",
)
