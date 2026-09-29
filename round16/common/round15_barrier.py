from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any, Mapping


_SHA256 = re.compile(r"^[0-9a-f]{64}$")
# This allow-list is deliberately exact.  A fuzzy ``"COMPLETE" in status`` gate
# would accept values such as ``ROUND15_NOT_COMPLETE`` and could start device
# validation before Round15 has released its resources.  If Round15 freezes a
# different terminal spelling, Round16 must review and add that exact value
# after reading the signed-off receipt.
_TERMINAL_STATUSES = frozenset(
    {
        "ROUND15_COMPLETE_GAIN",
        "ROUND15_COMPLETE_NO_NEW_GAIN",
        "ROUND15_MAINLINE_COMPLETE_GAIN",
        "ROUND15_MAINLINE_COMPLETE_NO_NEW_GAIN",
        "ROUND15_COMPLETE_SIMPLIFIED_DIRECTIONAL_NO_PROMOTION",
        "ROUND15_PARTIAL_WITH_GAPS",
        "ROUND15_BLOCKED_EXTERNAL_RESOURCE",
        "ROUND15_BLOCKED_TECHNICAL_CONTRACT",
    }
)


@dataclass(frozen=True)
class BarrierDecision:
    ready: bool
    reason_codes: tuple[str, ...]
    selected_baseline_sha256: str | None


def _is_terminal(status: object) -> bool:
    if not isinstance(status, str) or not status.strip():
        return False
    normalized = status.strip().upper()
    return normalized in _TERMINAL_STATUSES


def evaluate_round15_receipt(receipt: Mapping[str, Any]) -> BarrierDecision:
    reasons: list[str] = []

    if not _is_terminal(receipt.get("round15_terminal_status")):
        reasons.append("ROUND15_NOT_TERMINAL")

    baseline_sha = receipt.get("baseline_manifest_sha256")
    valid_baseline_sha = isinstance(baseline_sha, str) and bool(
        _SHA256.fullmatch(baseline_sha)
    )
    if not valid_baseline_sha:
        reasons.append("BASELINE_MANIFEST_IDENTITY_INVALID")

    dependency_identity = receipt.get("code_patch_dependency_identity")
    if not isinstance(dependency_identity, Mapping) or not dependency_identity:
        reasons.append("CODE_PATCH_DEPENDENCY_IDENTITY_MISSING")

    if receipt.get("cleanup_status") != "PASS":
        reasons.append("CLEANUP_NOT_PASS")

    owned_processes = receipt.get("owned_runtime_processes")
    if not isinstance(owned_processes, (list, tuple)) or owned_processes:
        reasons.append("ROUND15_RUNTIME_STILL_OWNED")

    if receipt.get("resource_lease_released") is not True:
        reasons.append("RESOURCE_LEASE_NOT_RELEASED")

    if receipt.get("baseline_correctness_open") is not False:
        reasons.append("BASELINE_CORRECTNESS_OPEN")

    return BarrierDecision(
        ready=not reasons,
        reason_codes=tuple(reasons),
        selected_baseline_sha256=baseline_sha if valid_baseline_sha else None,
    )
