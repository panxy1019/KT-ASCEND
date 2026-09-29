"""Fail-closed source-delta gate for the Round16 A1 observer overlay."""

from __future__ import annotations

import re
from typing import Any, Mapping

from .fresh_a1_profile import RUNTIME_VERSION

_SHA256 = re.compile(r"[0-9a-f]{64}")


class SourceDeltaError(RuntimeError):
    """A source file is neither the qualified parent nor an exact bound delta."""


def _exact_sha256(value: Any) -> bool:
    return isinstance(value, str) and _SHA256.fullmatch(value) is not None


def require_parent_or_bound_delta(
    *,
    relative: str,
    expected_parent_sha256: str,
    actual_sha256: str,
    overlay_receipt: Mapping[str, Any],
) -> str:
    """Accept a qualified parent byte identity or the single frozen A1 delta.

    The delta remains valid only while its overlay receipt says it is a
    default-off, observer-only change and binds both the original qualified
    file SHA and the installed file SHA.  Unknown paths fail closed.
    """
    if (
        not isinstance(relative, str)
        or not relative
        or not _exact_sha256(expected_parent_sha256)
        or not _exact_sha256(actual_sha256)
    ):
        raise SourceDeltaError("malformed source identity")
    if actual_sha256 == expected_parent_sha256:
        return "BASELINE_EXACT"
    if not isinstance(overlay_receipt, Mapping):
        raise SourceDeltaError("overlay receipt is unavailable")
    profile = overlay_receipt.get("round16_a1_profile")
    if (
        not isinstance(profile, Mapping)
        or profile.get("schema") != 1
        or profile.get("status") != "INSTALLED_DEFAULT_OFF_PROFILE_ONLY"
        or profile.get("runtime_version") != RUNTIME_VERSION
        or profile.get("default_enabled") is not False
        or profile.get("compute_semantics") != "UNCHANGED_OBSERVER_ONLY"
    ):
        raise SourceDeltaError("Round16 observer profile contract differs")
    parent = profile.get("parent_overlay_files")
    current = profile.get("modified_overlay_files")
    if (
        not isinstance(parent, Mapping)
        or not isinstance(current, Mapping)
        or set(parent) != set(current)
        or parent.get(relative) != expected_parent_sha256
        or current.get(relative) != actual_sha256
    ):
        raise SourceDeltaError("source delta is not exactly bound")
    return "ROUND16_BOUND_DELTA"


__all__ = (
    "SourceDeltaError",
    "require_parent_or_bound_delta",
)
