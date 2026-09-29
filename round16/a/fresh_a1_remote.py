"""Reviewed identities and host-side gates for fresh A1 device blocks.

Importing this module never runs a command.  The executable entry point is in
``tools/round16_a1_remote_block.py`` and requires a newly generated admission
receipt SHA on every invocation.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import time
from typing import Any, Mapping

from .fresh_a1_profile import (
    BLOCK_IDS,
    STAGE,
    runtime_container_name,
    runtime_evidence_name,
)


TRAINING_CONTAINER_ID = "0419bcfee46f317d6833e038c28e81cf3d3ecf47323a423062bac2764e0ab2ef"
REMOTE_BASE = Path("/mnt/data1/kt-round9")
V295_REMOTE = REMOTE_BASE / "round14a-a-gdn-model-20260927-v2/v295"
_TARGET = {
    "physical_id": 12,
    "card_id": 6,
    "chip_id": 0,
    "pci_bdf": "0000:85:00.0",
    "numa_nodes": [2, 3, 4, 7],
}
class RemoteControlError(RuntimeError):
    """A host-side launch, ownership, admission, or cleanup gate failed."""


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def block_runtime_contract(block: str) -> dict[str, Any]:
    if block not in BLOCK_IDS:
        raise ValueError("block must be exactly A or B")
    block_id = BLOCK_IDS[block]
    return {
        "schema": 1,
        "stage": STAGE,
        "block": block,
        "block_id": block_id,
        "container": runtime_container_name(block),
        "evidence": runtime_evidence_name(block),
        "model_start_limit": 1,
        "generation_limit": 6,
        "automatic_retries": 0,
        "abort_controls": 0,
        "profile_environment": {
            "ROUND16_A1_PROFILE_ENABLE": "1",
            "ROUND16_A1_BLOCK_ID": block_id,
            "ROUND16_A1_PROFILE_PATH": "/out/round16-a1-profile.jsonl",
        },
    }


def validate_admission(
    path: str | Path,
    *,
    expected_sha256: str,
    now: float | None = None,
    maximum_age_seconds: int = 600,
) -> dict[str, Any]:
    """Require exact bytes, physical12 identity, and a fresh launcher receipt."""
    receipt_path = Path(path)
    if _sha256(receipt_path) != expected_sha256:
        raise RemoteControlError("launcher admission SHA differs from reviewed input")
    try:
        value = json.loads(receipt_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise RemoteControlError(f"cannot read launcher admission receipt: {exc}") from exc
    if not isinstance(value, dict) or value.get("status") != "PASS":
        raise RemoteControlError("launcher admission status is not PASS")
    if value.get("target") != _TARGET:
        raise RemoteControlError("launcher admission target is not exact physical12")
    result_started = value.get("started_at")
    result_ended = value.get("ended_at")
    validated = value.get("validated_at_epoch")
    for label, observed in (
        ("started_at", result_started),
        ("ended_at", result_ended),
        ("validated_at_epoch", validated),
    ):
        if isinstance(observed, bool) or not isinstance(observed, (int, float)):
            raise RemoteControlError(f"launcher admission {label} is missing")
        if not math.isfinite(float(observed)):
            raise RemoteControlError(f"launcher admission {label} is nonfinite")
    current = time.time() if now is None else now
    age = current - float(validated)
    if age < 0 or age > maximum_age_seconds:
        raise RemoteControlError("launcher admission is stale or from the future")
    raw = value.get("raw_sha256")
    rounds = value.get("rounds")
    if (
        not isinstance(rounds, list)
        or len(rounds) != 2
        or any(
            not isinstance(name, str)
            or not name
            or Path(name).name != name
            or not name.endswith(".json")
            for name in rounds
        )
        or len(set(rounds)) != 2
    ):
        raise RemoteControlError("launcher admission round order is not two exact basenames")
    if not isinstance(raw, Mapping) or set(raw) != set(rounds):
        raise RemoteControlError("launcher admission raw round identities are incomplete")
    if list(raw) != rounds:
        raise RemoteControlError("launcher admission raw round order differs")
    if any(not isinstance(item, str) or len(item) != 64 for item in raw.values()):
        raise RemoteControlError("launcher admission raw round identities are incomplete")
    round_times: list[tuple[float, float]] = []
    for name in rounds:
        raw_path = receipt_path.parent / name
        if not raw_path.is_file() or _sha256(raw_path) != raw[name]:
            raise RemoteControlError(f"launcher admission raw round hash mismatch: {name}")
        try:
            document = json.loads(raw_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise RemoteControlError(f"cannot read launcher admission raw round: {exc}") from exc
        if not isinstance(document, dict):
            raise RemoteControlError("launcher admission raw round must be an object")
        started = document.get("started_at")
        ended = document.get("ended_at")
        if (
            isinstance(started, bool)
            or not isinstance(started, (int, float))
            or isinstance(ended, bool)
            or not isinstance(ended, (int, float))
            or not math.isfinite(float(started))
            or not math.isfinite(float(ended))
            or float(ended) - float(started) < 60.0
        ):
            raise RemoteControlError("each launcher admission raw round must span at least 60 seconds")
        round_times.append((float(started), float(ended)))
    if round_times[0][1] > round_times[1][0]:
        raise RemoteControlError("launcher admission raw round order overlaps or reverses")
    if (
        float(result_started) != round_times[0][0]
        or float(result_ended) != round_times[1][1]
        or float(validated) < float(result_ended)
    ):
        raise RemoteControlError("launcher admission wrapper times do not wrap the raw rounds")
    raw_age = current - float(result_ended)
    if raw_age < 0 or raw_age > maximum_age_seconds:
        raise RemoteControlError("launcher admission raw rounds are stale or from the future")
    return value


def launch_command(snapshot: str | Path, admission: str | Path) -> list[str]:
    root = Path(snapshot)
    return [
        "python3", str(root / "source/tools/launch_hybrid.py"),
        "--version", "295", "--storage", "MXFP4", "--placement", "static",
        "--resource-mode", "shared", "--shared-training-container", TRAINING_CONTAINER_ID,
        "--bank-receipt", str(root / "rebound/bank-receipt.json"),
        "--device-gate", str(REMOTE_BASE / "evidence/20260920-c128-expert-v19/result.json"),
        "--overlay-python", str(root / "overlay-python"),
        "--framework-profile", str(root / "framework-profile.json"),
        "--context-target", "4096",
        "--context-reference-profile", str(REMOTE_BASE / "evidence/20260922-plan10-m3-baseline-v64/context-reference-profile.json"),
        "--streaming-profile", str(root / "streaming-profile.json"),
        "--cancel-trace", "--cancel-closure", "--cost-mode", "OFF",
        "--cost-manifest", str(root / "cost-manifest.json"),
        "--cost-budget", str(root / "execution-budget.json"),
        "--cost-admission", str(Path(admission)),
    ]


__all__ = (
    "REMOTE_BASE",
    "RemoteControlError",
    "TRAINING_CONTAINER_ID",
    "V295_REMOTE",
    "block_runtime_contract",
    "launch_command",
    "validate_admission",
)
