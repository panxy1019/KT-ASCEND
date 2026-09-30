"""Post-load binding of KT complements to untouched native MoE runners.

This module stays framework-import free.  The live worker injects the exact
frozen selector module and per-layer callbacks after checkpoint loading, while
the binding enforces that every routed expert still belongs to the concrete
``AscendMoERunner`` created by the Ascend factory.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Sequence

from round16.b.selective_loader import SelectiveLoadPlan
from round18.native_hybrid_quant_method import NativeHybridQuantMethodAdapter
from round18.native_runner_seam import NATIVE_PARTIAL_SEAM_API


NATIVE_HYBRID_LIVE_BINDING_API = (
    "round18:native-runner-postload-kt-complement:v1"
)


@dataclass(frozen=True)
class NativeHybridLayerBinding:
    layer_name: str
    runner_class: str
    delegate_class: str
    physical_slots: int
    runner_preserved: bool


def attach_native_hybrid_quant_methods(
    *,
    resolved_layers: Sequence[tuple[str, Any]],
    plan: SelectiveLoadPlan,
    selector_module: Any,
    callbacks_for_layer: Callable[
        [str], tuple[Callable[..., Any], Callable[..., Any], Callable[..., Any]]
    ],
    identity_provider_for_layer: Callable[
        [str], tuple[Callable[..., Any], Callable[[tuple[Any, ...]], bool]]
    ],
    expected_layer_count: int,
    route_observer: Callable[..., Any] | None = None,
) -> tuple[NativeHybridLayerBinding, ...]:
    """Wrap each native quant method without changing its runner or owner.

    The operation validates the complete layer set before committing any
    wrapper, so a structural mismatch cannot leave a partially bound model.
    Callback construction may initialize the qualified CPU layer wrapper; it
    happens only after all native runner/owner invariants have passed.
    """

    if not isinstance(plan, SelectiveLoadPlan):
        raise TypeError("live native Hybrid binding requires SelectiveLoadPlan")
    if plan.logical_expert_count != 512 or plan.physical_expert_count != 32:
        raise ValueError("live native Hybrid binding requires logical512/Q32")
    if type(expected_layer_count) is not int or expected_layer_count <= 0:
        raise ValueError("expected layer count must be positive")
    if not callable(callbacks_for_layer) or not callable(
        identity_provider_for_layer
    ):
        raise TypeError("live native Hybrid binding callbacks are incomplete")
    if not callable(getattr(selector_module, "select_experts", None)):
        raise TypeError("live native Hybrid binding lacks the native selector")

    layers = tuple(resolved_layers)
    if len(layers) != expected_layer_count:
        raise RuntimeError(
            f"expected {expected_layer_count} native layers, observed {len(layers)}"
        )
    prepared: list[tuple[str, Any, Any, type]] = []
    names: set[str] = set()
    for layer_name, runner in layers:
        if type(layer_name) is not str or not layer_name or layer_name in names:
            raise RuntimeError("native Hybrid layer names must be unique")
        names.add(layer_name)
        runner_class = type(runner)
        if "AscendMoERunner" not in runner_class.__name__:
            raise TypeError("Round18 may bind only the native AscendMoERunner")
        if getattr(runner, "round18_native_partial_api", None) != (
            NATIVE_PARTIAL_SEAM_API
        ):
            raise ValueError("native runner lacks the qualified Q32 partial seam")
        if getattr(runner, "round18_selective_plan", None) is not plan:
            raise ValueError("native runner and live binding use different Q32 plans")
        if getattr(runner, "round18_native_runner_class", None) is not runner_class:
            raise ValueError("native runner construction identity was not retained")
        owner = getattr(runner, "routed_experts", None)
        if owner is None or getattr(owner, "round16_selective_plan", None) is not plan:
            raise ValueError("native runner Q32 owner identity differs")
        delegate = getattr(owner, "quant_method", None)
        if delegate is None or not callable(getattr(delegate, "apply", None)):
            raise TypeError("native Q32 owner lacks its Ascend quant method")
        prepared.append((layer_name, runner, delegate, runner_class))

    adapters: list[tuple[Any, NativeHybridQuantMethodAdapter, type]] = []
    receipts: list[NativeHybridLayerBinding] = []
    for layer_name, runner, delegate, runner_class in prepared:
        submit, sync, abort = callbacks_for_layer(layer_name)
        identity_provider, identity_is_current = identity_provider_for_layer(
            layer_name
        )
        adapter = NativeHybridQuantMethodAdapter(
            delegate=delegate,
            selector_module=selector_module,
            layer_name=layer_name,
            identity_provider=identity_provider,
            cpu_submit=submit,
            cpu_sync=sync,
            cpu_abort=abort,
            identity_is_current=identity_is_current,
            route_observer=route_observer,
        )
        adapters.append((runner.routed_experts, adapter, runner_class))
        receipts.append(
            NativeHybridLayerBinding(
                layer_name=layer_name,
                runner_class=runner_class.__name__,
                delegate_class=type(delegate).__name__,
                physical_slots=32,
                runner_preserved=True,
            )
        )

    for owner, adapter, _runner_class in adapters:
        # ``RoutedExperts`` is an nn.Module and the frozen Ascend method is an
        # already-registered child module.  Replacing that child with a plain
        # adapter through nn.Module.__setattr__ is both illegal and would alter
        # state/load ownership.  A post-load instance shadow keeps the exact
        # native method registered in ``_modules`` while the runner's
        # ``_quant_method`` property resolves the Round18 apply proxy.
        registered = getattr(owner, "_modules", {}).get("quant_method")
        if registered is not None and registered is not adapter.delegate:
            raise RuntimeError("registered native quant method identity changed")
        owner.__dict__["quant_method"] = adapter
        owner.__dict__["round18_registered_native_quant_method"] = adapter.delegate
        owner.round18_native_hybrid_api = NATIVE_HYBRID_LIVE_BINDING_API
    for (_layer_name, runner, _delegate, runner_class), (
        _owner,
        adapter,
        _expected_class,
    ) in zip(prepared, adapters, strict=True):
        if type(runner) is not runner_class:
            raise RuntimeError("native runner class changed during live binding")
        if runner.routed_experts.quant_method is not adapter:
            raise RuntimeError("native runner did not retain the installed adapter")
        runner.round18_native_hybrid_api = NATIVE_HYBRID_LIVE_BINDING_API

    return tuple(receipts)


__all__ = (
    "NATIVE_HYBRID_LIVE_BINDING_API",
    "NativeHybridLayerBinding",
    "attach_native_hybrid_quant_methods",
)
