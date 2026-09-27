"""Pre-dispatch H-BOUNDED request gate for the Round14 narrow profile."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import time


_CACHE = None


def parse_profile(document: dict) -> dict:
    required = {
        "schema", "mode", "max_input_tokens", "max_output_tokens",
        "max_total_tokens", "hybrid_threshold_tokens", "outside_policy",
    }
    if (
        not isinstance(document, dict)
        or set(document) != required
        or document.get("schema") != 1
        or document.get("mode") != "H_BOUNDED"
        or document.get("outside_policy") != "REJECT"
        or any(type(document.get(key)) is not int or document[key] < 1 for key in (
            "max_input_tokens", "max_output_tokens", "max_total_tokens",
            "hybrid_threshold_tokens",
        ))
        or document["hybrid_threshold_tokens"] != document["max_input_tokens"] + 1
        or document["max_total_tokens"] < document["max_input_tokens"]
        or document["max_total_tokens"] > document["max_input_tokens"] + document["max_output_tokens"]
    ):
        raise ValueError("invalid Round14 H-BOUNDED profile")
    return dict(document)


def _emit(path: Path, row: dict) -> None:
    path = Path(path)
    if path.exists() and path.stat().st_size > 8 * 1024**2:
        raise RuntimeError("Round14 H-BOUNDED event ceiling exceeded")
    with path.open("a") as stream:
        stream.write(json.dumps(row, allow_nan=False, separators=(",", ":")) + "\n")


def enforce(profile: dict, obj, input_ids, *, event_path=Path("/out/round14-boundary-events.jsonl")):
    profile = parse_profile(profile)
    if not isinstance(input_ids, list) or any(type(token) is not int for token in input_ids):
        raise ValueError("H-BOUNDED requires concrete token IDs")
    sampling = getattr(obj, "sampling_params", None)
    output = sampling.get("max_new_tokens") if isinstance(sampling, dict) else None
    if type(output) is not int or output < 1:
        raise ValueError("H-BOUNDED requires a positive max_new_tokens")
    inputs = len(input_ids)
    total = inputs + output
    reasons = []
    if inputs > profile["max_input_tokens"]:
        reasons.append("INPUT_LIMIT")
    if output > profile["max_output_tokens"]:
        reasons.append("OUTPUT_LIMIT")
    if total > profile["max_total_tokens"]:
        reasons.append("TOTAL_CONTEXT_LIMIT")
    row = {
        "status": "ROUND14_H_BOUNDED_REJECT" if reasons else "ROUND14_H_BOUNDED_ACCEPT",
        "time": time.time(),
        "request_id": getattr(obj, "rid", None),
        "input_tokens": inputs,
        "max_new_tokens": output,
        "total_tokens": total,
        "limits": {
            "max_input_tokens": profile["max_input_tokens"],
            "max_output_tokens": profile["max_output_tokens"],
            "max_total_tokens": profile["max_total_tokens"],
            "hybrid_threshold_tokens": profile["hybrid_threshold_tokens"],
        },
        "reasons": reasons,
        "before_scheduler_dispatch": True,
    }
    _emit(Path(event_path), row)
    if reasons:
        raise ValueError(
            "H-BOUNDED rejected request before scheduler dispatch: " + ",".join(reasons)
        )
    return True


def _load_environment_profile():
    global _CACHE
    path_value = os.environ.get("ROUND14_H_BOUNDED_PROFILE")
    if not path_value:
        return None
    expected = os.environ.get("ROUND14_H_BOUNDED_PROFILE_SHA256")
    if not expected or len(expected) != 64:
        raise RuntimeError("Round14 H-BOUNDED profile hash is missing")
    path = Path(path_value).resolve(strict=True)
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != expected:
        raise RuntimeError("Round14 H-BOUNDED profile hash mismatch")
    key = (str(path), actual)
    if _CACHE is None or _CACHE[0] != key:
        _CACHE = (key, parse_profile(json.loads(path.read_text())))
    return _CACHE[1]


def enforce_from_environment(obj, input_ids):
    profile = _load_environment_profile()
    if profile is None:
        return None
    event = Path(os.environ.get(
        "ROUND14_H_BOUNDED_EVENTS", "/out/round14-boundary-events.jsonl"
    ))
    return enforce(profile, obj, input_ids, event_path=event)


def reset_for_test():
    global _CACHE
    _CACHE = None
