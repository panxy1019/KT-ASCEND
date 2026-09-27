#!/usr/bin/env python3
"""Create the exact tied-parameter-aware Round12a OffloaderV1 source."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from typing import TypeVar


BASE_OFFLOADER_SHA256 = (
    "98a37555dd1fb6bda68dc5485ca0999dce75a5c43a6df62f86a16f740532442e"
)

_T = TypeVar("_T")


def unique_items_by_identity(state: Mapping[str, _T]) -> list[tuple[str, _T]]:
    """Keep the first registered name for each exact tensor object."""
    seen: set[int] = set()
    unique: list[tuple[str, _T]] = []
    for name, value in state.items():
        identity = id(value)
        if identity in seen:
            continue
        seen.add(identity)
        unique.append((name, value))
    return unique


_HELPER = '''

def _device_state_preserving_ties(module: torch.nn.Module, device: torch.device):
    """Copy each exact state object once and let functional_call retain ties."""
    state = module.state_dict(keep_vars=True)
    seen = set()
    device_state = {}
    for name, value in state.items():
        identity = id(value)
        if identity in seen:
            continue
        seen.add(identity)
        device_state[name] = value.to(device, non_blocking=True)
    return device_state
'''

_OLD_DEVICE_STATE = '''                device_state = {
                    # here we blindly call `to(device)`
                    # if the parameter is already on the device, it will be a no-op
                    k: v.to(device, non_blocking=True)
                    for k, v in module.state_dict().items()
                }'''

_NEW_DEVICE_STATE = '''                device_state = _device_state_preserving_ties(module, device)'''


def patch_offloader_source(source: str) -> str:
    """Patch only OffloaderV1's device-state construction, failing on drift."""
    digest = hashlib.sha256(source.encode()).hexdigest()
    if digest != BASE_OFFLOADER_SHA256:
        raise ValueError(f"OffloaderV1 base source identity mismatch: {digest}")
    class_anchor = "\n\nclass OffloaderV1(BaseOffloader):"
    if source.count(class_anchor) != 1:
        raise ValueError("OffloaderV1 class anchor mismatch")
    if source.count(_OLD_DEVICE_STATE) != 1:
        raise ValueError("OffloaderV1 device-state anchor mismatch")
    patched = source.replace(class_anchor, _HELPER + class_anchor)
    patched = patched.replace(_OLD_DEVICE_STATE, _NEW_DEVICE_STATE)
    compile(patched, "offloader.py", "exec")
    return patched


def candidate_sha256(source: str) -> str:
    return hashlib.sha256(patch_offloader_source(source).encode()).hexdigest()
