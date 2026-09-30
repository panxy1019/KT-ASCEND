"""Minimal CPU complement around the frozen native Ascend quant method.

The adapter deliberately leaves ``AscendMoERunner`` in place.  It observes
the exact selector call already made by the frozen quant method, submits the
same logical Top-K to KT, lets the native method compute the resident partial,
then adds the already-weighted CPU partial to ``FusedExpertsResult.routed_out``.
Shared experts, routed scaling, communication finalize and residual handling
remain outside this boundary and therefore stay owned by the native runner.
"""

from __future__ import annotations

from threading import RLock
from typing import Any, Callable


NATIVE_HYBRID_QUANT_METHOD_API = (
    "round18:native-ascend-quant-method-with-kt-complement:v1"
)
_SELECTOR_PATCH_LOCK = RLock()


class StaleHybridWorkError(RuntimeError):
    """The CPU result was drained but belongs to an obsolete request epoch."""


class NativeHybridQuantMethodAdapter:
    """Wrap one native routed-expert quant method without replacing its runner.

    The frozen native method resolves ``select_experts`` through its module
    global.  A process-wide lock makes the observation scope atomic for the
    Round18 TP1/single-request qualification.  The original selector itself is
    still called exactly once and its tensors are returned unchanged.
    """

    round18_api_identity = NATIVE_HYBRID_QUANT_METHOD_API

    def __init__(
        self,
        *,
        delegate: Any,
        selector_module: Any,
        layer_name: str,
        identity_provider: Callable[..., Any],
        cpu_submit: Callable[..., Any],
        cpu_sync: Callable[..., Any],
        cpu_abort: Callable[..., Any],
        identity_is_current: Callable[[tuple[Any, ...]], bool],
        route_observer: Callable[..., Any] | None = None,
    ) -> None:
        if delegate is None or not callable(getattr(delegate, "apply", None)):
            raise TypeError("native quant-method delegate must provide apply")
        selector = getattr(selector_module, "select_experts", None)
        if not callable(selector):
            raise TypeError("selector module must expose select_experts")
        if type(layer_name) is not str or not layer_name:
            raise ValueError("layer_name must be non-empty")
        callbacks = {
            "identity_provider": identity_provider,
            "cpu_submit": cpu_submit,
            "cpu_sync": cpu_sync,
            "cpu_abort": cpu_abort,
            "identity_is_current": identity_is_current,
        }
        for name, callback in callbacks.items():
            if not callable(callback):
                raise TypeError(f"{name} must be callable")

        self.delegate = delegate
        self.selector_module = selector_module
        self.layer_name = layer_name
        self.identity_provider = identity_provider
        self.cpu_submit = cpu_submit
        self.cpu_sync = cpu_sync
        self.cpu_abort = cpu_abort
        self.identity_is_current = identity_is_current
        if route_observer is not None and not callable(route_observer):
            raise TypeError("route_observer must be callable")
        self.route_observer = route_observer
        self._original_selector = selector
        self._generation = 0
        self._completed_generations = 0
        self._active = False

    @property
    def completed_generations(self) -> int:
        return self._completed_generations

    def __getattr__(self, name: str) -> Any:
        # Preserve the exact native method's non-apply interface.  Production
        # installation happens after post-load, but this keeps source and
        # diagnostic attributes inspectable without copying the method.
        return getattr(self.delegate, name)

    @staticmethod
    def _identity_tuple(value: Any) -> tuple[Any, ...]:
        if isinstance(value, tuple):
            result = value
        else:
            try:
                result = tuple(value)
            except TypeError as error:
                raise TypeError("work identity must be tuple-compatible") from error
        if not result:
            raise ValueError("work identity must be non-empty")
        return result

    @staticmethod
    def _cpu_output(partial: Any) -> Any:
        if getattr(partial, "route_weights_applied", None) is not True:
            raise ValueError("CPU partial must contain original Top-K weights")
        if getattr(partial, "routed_scaling_applied", None) is not False:
            raise ValueError("CPU partial must exclude routed scaling")
        if not hasattr(partial, "output"):
            raise TypeError("CPU completion does not expose its weighted partial")
        return partial.output

    def apply(self, *args: Any, **kwargs: Any) -> Any:
        if "x" not in kwargs or "router_logits" not in kwargs:
            raise TypeError("Round18 native Hybrid apply requires keyword x/router_logits")

        with _SELECTOR_PATCH_LOCK:
            if self._active:
                raise RuntimeError("native Hybrid adapter is already active")
            if self.selector_module.select_experts is not self._original_selector:
                raise RuntimeError("native selector identity changed before execution")

            self._active = True
            self._generation += 1
            generation = self._generation
            selector_calls = 0
            pending = None
            identity_key: tuple[Any, ...] | None = None

            def observe_selector(*selector_args: Any, **selector_kwargs: Any) -> Any:
                nonlocal selector_calls, pending, identity_key
                selector_calls += 1
                if selector_calls != 1:
                    raise RuntimeError("native selector must execute exactly once")
                topk_weights, topk_ids = self._original_selector(
                    *selector_args, **selector_kwargs
                )
                hidden_states = selector_kwargs.get("hidden_states")
                if hidden_states is None and selector_args:
                    hidden_states = selector_args[0]
                identity_key = self._identity_tuple(
                    self.identity_provider(
                        layer_name=self.layer_name,
                        generation=generation,
                        hidden_states=hidden_states,
                        topk_ids=topk_ids,
                        topk_weights=topk_weights,
                    )
                )
                if self.route_observer is not None:
                    self.route_observer(
                        layer_name=self.layer_name,
                        generation=generation,
                        hidden_states=hidden_states,
                        topk_ids=topk_ids,
                        topk_weights=topk_weights,
                        identity_key=identity_key,
                    )
                pending = self.cpu_submit(
                    hidden_states=hidden_states,
                    topk_ids=topk_ids,
                    topk_weights=topk_weights,
                    identity_key=identity_key,
                )
                return topk_weights, topk_ids

            self.selector_module.select_experts = observe_selector
            try:
                native_result = self.delegate.apply(*args, **kwargs)
                if selector_calls != 1 or pending is None or identity_key is None:
                    raise RuntimeError("native selector must execute exactly once")
            except BaseException:
                if pending is not None and identity_key is not None:
                    self.cpu_abort(
                        pending=pending,
                        identity_key=identity_key,
                        reference_tensor=kwargs.get("x"),
                    )
                raise
            finally:
                self.selector_module.select_experts = self._original_selector
                self._active = False

            if not hasattr(native_result, "routed_out"):
                # The CPU task is still outstanding, so drain it before
                # rejecting an incompatible native return contract.
                self.cpu_abort(
                    pending=pending,
                    identity_key=identity_key,
                    reference_tensor=kwargs.get("x"),
                )
                raise TypeError("native quant method did not return FusedExpertsResult")

            cpu_partial = self.cpu_sync(
                pending=pending,
                reference_tensor=native_result.routed_out,
                identity_key=identity_key,
            )
            cpu_output = self._cpu_output(cpu_partial)
            if not self.identity_is_current(identity_key):
                raise StaleHybridWorkError(
                    "drained CPU result belongs to a stale request epoch"
                )
            native_result.routed_out = native_result.routed_out + cpu_output
            self._completed_generations += 1
            return native_result


__all__ = (
    "NATIVE_HYBRID_QUANT_METHOD_API",
    "NativeHybridQuantMethodAdapter",
    "StaleHybridWorkError",
)
