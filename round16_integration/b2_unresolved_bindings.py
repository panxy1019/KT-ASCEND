"""Executable fail-closed factories for the first donor device handoff.

The module exists so operational commands never refer to a fictional factory.
It deliberately raises before model construction or device work until the
donor image supplies and verifies the physical-Q32 RoutedExperts allocation
and checkpoint-slice loader required by the B2 plugin.
"""

from __future__ import annotations

from typing import Any

from round16_integration.b2_worker_overlay_entry import B2_BINDINGS_FACTORY_API
from round16_integration.p1_allgather_q32_probe import P1_RUNTIME_FACTORY_API


UNRESOLVED_MESSAGE = (
    "SELECTIVE_Q32_PREALLOCATION_REQUIRED: bind a device-verified "
    "RoutedExperts class that allocates physical32 before create_weights and "
    "loads only the frozen Q32 checkpoint slices; no model or NPU work ran"
)


def create_b2_bindings(**_kwargs: Any) -> Any:
    raise RuntimeError(UNRESOLVED_MESSAGE)


create_b2_bindings.round16_api_identity = B2_BINDINGS_FACTORY_API


def create_p1_runtime_binding(_context: Any) -> Any:
    raise RuntimeError(UNRESOLVED_MESSAGE)


create_p1_runtime_binding.round16_api_identity = P1_RUNTIME_FACTORY_API


__all__ = (
    "UNRESOLVED_MESSAGE",
    "create_b2_bindings",
    "create_p1_runtime_binding",
)
