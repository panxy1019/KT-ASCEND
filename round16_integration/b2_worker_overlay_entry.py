"""Default-OFF donor worker bootstrap for the exact B2 plugin.

The applicable donor patch calls :func:`maybe_install_from_environment` once
immediately before ``NPUWorker`` invokes ``self.model_runner.load_model()``.
No runtime module is imported and no class is patched unless
``ROUND16_B_ENABLE_B2=1`` is set explicitly.
"""

from __future__ import annotations

from dataclasses import dataclass
import importlib
import json
import os
from pathlib import Path
from typing import Any, Callable, Mapping

from round16.b.exact_donor_worker_hook import (
    ExactWorkerHookHandle,
    ExactWorkerRuntimeRegistry,
    ExactWorkerSourceToken,
    load_exact_worker_source_receipt,
)
from round16_integration.b2_exact_worker_plugin import (
    B2ExactWorkerPluginBindings,
    install_b2_exact_worker_plugin,
)


B2_BINDINGS_FACTORY_API = "round16-integration:b2-bindings-factory:v1"


@dataclass(frozen=True)
class InstalledB2WorkerPlugin:
    handle: ExactWorkerHookHandle
    registry: ExactWorkerRuntimeRegistry
    factory_specification: str


def _load_factory(specification: str) -> Callable[..., B2ExactWorkerPluginBindings]:
    if type(specification) is not str or specification.count(":") != 1:
        raise ValueError("B2 bindings factory must use module:callable syntax")
    module_name, attribute = specification.split(":", 1)
    if not module_name or not attribute:
        raise ValueError("B2 bindings factory must use module:callable syntax")
    factory = getattr(importlib.import_module(module_name), attribute)
    if not callable(factory):
        raise TypeError("B2 bindings factory must be callable")
    if getattr(factory, "round16_api_identity", None) != B2_BINDINGS_FACTORY_API:
        raise ValueError("B2 bindings factory identity is not verified")
    return factory


def maybe_install_from_environment(
    model_runner_cls: type,
    *,
    environment: Mapping[str, str] | None = None,
) -> InstalledB2WorkerPlugin | None:
    """Install from an explicit source receipt and donor binding factory."""

    values = os.environ if environment is None else environment
    enabled = values.get("ROUND16_B_ENABLE_B2")
    if enabled in (None, "", "0"):
        return None
    if enabled != "1":
        raise ValueError("ROUND16_B_ENABLE_B2 must be exactly 1 when enabled")
    receipt_value = values.get("ROUND16_B2_WORKER_PREFLIGHT")
    if type(receipt_value) is not str or not receipt_value:
        raise ValueError("ROUND16_B2_WORKER_PREFLIGHT is required")
    receipt_path = Path(receipt_value).resolve()
    if not receipt_path.is_file() or receipt_path.stat().st_size > 1024 * 1024:
        raise ValueError("B2 worker preflight receipt is unavailable or too large")
    payload = json.loads(receipt_path.read_text())
    worker_token: ExactWorkerSourceToken = load_exact_worker_source_receipt(
        payload
    )
    specification = values.get("ROUND16_B2_BINDINGS_FACTORY")
    factory = _load_factory(specification)
    bindings = factory(
        model_runner_cls=model_runner_cls,
        worker_source_token=worker_token,
    )
    if not isinstance(bindings, B2ExactWorkerPluginBindings):
        raise TypeError("B2 bindings factory returned the wrong type")
    if bindings.model_runner_cls is not model_runner_cls:
        raise ValueError("B2 bindings factory changed model runner identity")
    if bindings.worker_source_token != worker_token:
        raise ValueError("B2 bindings factory changed worker source token")
    handle, registry = install_b2_exact_worker_plugin(bindings, enabled=True)
    if registry is None or handle.installed is not True:
        raise RuntimeError("B2 exact worker hook did not install")
    return InstalledB2WorkerPlugin(handle, registry, specification)


__all__ = (
    "B2_BINDINGS_FACTORY_API",
    "InstalledB2WorkerPlugin",
    "maybe_install_from_environment",
)
