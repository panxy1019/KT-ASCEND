"""Explicit, idempotent registration without importing SGLang."""

from __future__ import annotations

from collections.abc import MutableMapping
from dataclasses import dataclass
from typing import Any

from round16.a.provider_manifest import PROVIDER_NAME
from round16.a.sglang_adapter import KTAscendProvider


class RegistrationConflictError(RuntimeError):
    pass


@dataclass(frozen=True)
class RegistrationReceipt:
    provider_name: str
    created: bool


def register_provider(
    registry: Any,
    *,
    provider_factory: Any = KTAscendProvider,
) -> RegistrationReceipt:
    """Register once in a mapping or a registry with get/register methods."""
    if isinstance(registry, MutableMapping):
        existing = registry.get(PROVIDER_NAME)
        if existing is None:
            registry[PROVIDER_NAME] = provider_factory
            return RegistrationReceipt(PROVIDER_NAME, True)
        if existing is provider_factory:
            return RegistrationReceipt(PROVIDER_NAME, False)
        raise RegistrationConflictError(
            f"{PROVIDER_NAME!r} is already bound to a different provider"
        )

    getter = getattr(registry, "get_provider", None)
    registrar = getattr(registry, "register_provider", None)
    if getter is None or registrar is None:
        raise TypeError(
            "registry must be a mutable mapping or expose get_provider/register_provider"
        )
    existing = getter(PROVIDER_NAME)
    if existing is None:
        registrar(PROVIDER_NAME, provider_factory)
        return RegistrationReceipt(PROVIDER_NAME, True)
    if existing is provider_factory:
        return RegistrationReceipt(PROVIDER_NAME, False)
    raise RegistrationConflictError(
        f"{PROVIDER_NAME!r} is already bound to a different provider"
    )
