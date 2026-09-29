"""Default-off runtime observer copied into a fresh A1 snapshot.

Only the explicitly enabled path imports ``torch`` and creates device events.
The observer never calls a device synchronization primitive.  Native CPU
expert execution stays UNKNOWN because the existing submit/sync API has no
worker start/end callback.
"""

from __future__ import annotations

import functools
import json
import os
import re
import time
from types import MethodType
from typing import Any, Callable


_ENABLE_ENV = "ROUND16_A1_PROFILE_ENABLE"
_BLOCK_ENV = "ROUND16_A1_BLOCK_ID"
_PATH_ENV = "ROUND16_A1_PROFILE_PATH"
_BINDINGS_ENV = "ROUND16_A1_BINDINGS_PATH"
_BLOCK_IDS = frozenset(
    {
        "round16-a1-fresh-profile-a-v001",
        "round16-a1-fresh-profile-b-v001",
    }
)
_HOST_LAYERS = frozenset(range(48))
_DEVICE_SAMPLE_LAYERS = frozenset({0, 24, 47})
_LAYER_PATTERN = re.compile(r"(?:^|\.)layers\.(\d+)(?:\.|$)")

_ACTIVE_STATE: "_State | None" = None
_PENDING_STATES: list["_State"] = []
_BATCH_COUNTER = 0


def _enabled() -> bool:
    return os.environ.get(_ENABLE_ENV) == "1"


class _State:
    def __init__(
        self,
        runner: object,
        block_id: str,
        profile_path: str,
        batch: int,
        request_id: str,
    ) -> None:
        self.runner = runner
        self.block_id = block_id
        self.profile_path = profile_path
        self.batch = batch
        self.request_id = request_id
        self.batch_span_id = f"{block_id}:batch:{batch}"
        self.serial = 0
        self.rows: list[dict[str, Any]] = []
        self.pending: list[tuple[object, object, dict[str, Any]]] = []
        self.stack: list[tuple[str, str]] = []

    def identity(
        self,
        component: str,
        layer: int,
        *,
        parent_span_id: str | None = None,
        correlation_id: str | None = None,
        stream_id: str = "host-main",
    ) -> dict[str, Any]:
        self.serial += 1
        stem = f"{self.block_id}:b{self.batch}:l{layer}:s{self.serial}"
        return {
            "schema": 1,
            "block_id": self.block_id,
            "batch_ordinal": self.batch,
            "request_id": self.request_id,
            "layer": layer,
            "component": component,
            "span_id": stem,
            "parent_span_id": parent_span_id or self.batch_span_id,
            "stream_id": stream_id,
            "start_event_id": f"{stem}:start",
            "end_event_id": f"{stem}:end",
            "correlation_id": correlation_id or f"{stem}:correlation",
        }


def _active_for_layer(layer: int, *, device_sample: bool = False) -> _State | None:
    state = _ACTIVE_STATE
    layers = _DEVICE_SAMPLE_LAYERS if device_sample else _HOST_LAYERS
    if not _enabled() or state is None or layer not in layers:
        return None
    return state


def _parent_context(state: _State) -> tuple[str, str]:
    if state.stack:
        return state.stack[-1]
    return state.batch_span_id, f"{state.batch_span_id}:correlation"


def _host_call(
    state: _State,
    *,
    component: str,
    layer: int,
    operation: Callable[..., Any],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    parent_span_id: str,
    correlation_id: str,
    identity: dict[str, Any] | None = None,
) -> Any:
    if identity is None:
        identity = state.identity(
            component,
            layer,
            parent_span_id=parent_span_id,
            correlation_id=correlation_id,
        )
    try:
        started = time.monotonic_ns()
    except Exception:
        return operation(*args, **kwargs)
    outcome = "error"
    try:
        result = operation(*args, **kwargs)
        outcome = "return"
        return result
    finally:
        try:
            ended = time.monotonic_ns()
            if type(started) is int and type(ended) is int and ended >= started:
                state.rows.append(
                    {
                        **identity,
                        "clock_domain": "HOST_MONOTONIC_NS",
                        "start_ns": started,
                        "end_ns": ended,
                        "duration_ns": ended - started,
                        "elapsed_ms": None,
                        "outcome": outcome,
                        "status": "COMPLETE",
                        "derivation": "DIRECT_OBSERVATION",
                    }
                )
        except Exception:
            pass


def _cpu_unknown(
    state: _State,
    *,
    layer: int,
    parent_span_id: str,
    correlation_id: str,
) -> None:
    try:
        identity = state.identity(
            "CPU_EXPERT",
            layer,
            parent_span_id=parent_span_id,
            correlation_id=correlation_id,
            stream_id="native-cpu-worker-unobservable",
        )
        state.rows.append(
            {
                **identity,
                "clock_domain": "NATIVE_WORKER_CLOCK_UNAVAILABLE",
                "start_ns": None,
                "end_ns": None,
                "duration_ns": None,
                "status": "UNKNOWN_NATIVE_ABI_NO_CALLBACK",
                "reason": (
                    "submit_forward/sync_forward expose no native worker "
                    "start/end callback"
                ),
                "derivation": "NOT_INFERRED_FROM_HOST_BOOKENDS",
            }
        )
    except Exception:
        pass


def _wrap_method(method: object, name: str, component: str, layer: int) -> None:
    original = getattr(method, name)

    @functools.wraps(original)
    def wrapped(_self: object, *args: Any, **kwargs: Any) -> Any:
        state = _active_for_layer(layer)
        if state is None:
            return original(*args, **kwargs)
        parent_span_id, correlation_id = _parent_context(state)
        return _host_call(
            state,
            component=component,
            layer=layer,
            operation=original,
            args=args,
            kwargs=kwargs,
            parent_span_id=parent_span_id,
            correlation_id=correlation_id,
        )

    setattr(method, name, MethodType(wrapped, method))


def _wrap_parent(method: object, layer: int) -> None:
    original = getattr(method, "apply")

    @functools.wraps(original)
    def wrapped(_self: object, *args: Any, **kwargs: Any) -> Any:
        state = _active_for_layer(layer)
        if state is None:
            return original(*args, **kwargs)
        identity = state.identity("PARENT_ROUTED_EXPERT", layer)
        state.stack.append((identity["span_id"], identity["correlation_id"]))
        try:
            return _host_call(
                state,
                component="PARENT_ROUTED_EXPERT",
                layer=layer,
                operation=original,
                args=args,
                kwargs=kwargs,
                parent_span_id=identity["parent_span_id"],
                correlation_id=identity["correlation_id"],
                identity=identity,
            )
        finally:
            _cpu_unknown(
                state,
                layer=layer,
                parent_span_id=identity["span_id"],
                correlation_id=identity["correlation_id"],
            )
            if state.stack:
                state.stack.pop()

    setattr(method, "apply", MethodType(wrapped, method))


def _bind_runner(runner: object) -> None:
    worker = getattr(runner, "tp_worker", runner)
    model_runner = getattr(worker, "model_runner")
    model = getattr(model_runner, "model")
    bound: list[int] = []
    for name, module in model.named_modules():
        match = _LAYER_PATTERN.search(name)
        if match is None or int(match.group(1)) not in _HOST_LAYERS:
            continue
        method = getattr(module, "quant_method", None)
        if method is None and module.__class__.__name__ == "KTEPWrapperMethod":
            method = module
        if method.__class__.__name__ != "KTEPWrapperMethod":
            continue
        if getattr(method, "_round16_a1_observer_bound", False):
            bound.append(int(match.group(1)))
            continue
        layer = int(match.group(1))
        _wrap_method(method, "_submit_cpu_forward", "HOST_DISPATCH", layer)
        _wrap_method(method, "_sync_cpu_forward", "HOST_SYNC", layer)
        _wrap_parent(method, layer)
        setattr(method, "_round16_a1_observer_bound", True)
        bound.append(layer)
    if set(bound) != _HOST_LAYERS:
        raise RuntimeError("Round16 A1 Host expert binding coverage differs from layers 0..47")
    bindings_path = os.environ.get(_BINDINGS_ENV, "/out/round16-a1-bindings.json")
    path = os.fspath(bindings_path)
    if not os.path.exists(path):
        with open(path, "x", encoding="utf-8") as destination:
            json.dump(
                {
                    "schema": 1,
                    "status": "ROUND16_A1_OBSERVER_BOUND",
                    "host_layers": sorted(_HOST_LAYERS),
                    "device_sample_layers": sorted(_DEVICE_SAMPLE_LAYERS),
                    "host_components": [
                        "PARENT_ROUTED_EXPERT",
                        "HOST_DISPATCH",
                        "HOST_SYNC",
                        "HOST_MERGE",
                    ],
                    "device_query": "DEFERRED_QUERY_ONLY_NO_SYNCHRONIZE",
                    "cpu_expert": "UNKNOWN_NATIVE_ABI_NO_CALLBACK",
                },
                destination,
                sort_keys=True,
            )


def before(runner: object, batch: object) -> None:
    """Start one scheduler-batch observation; disabled path is a direct no-op."""
    if not _enabled():
        return
    block_id = os.environ.get(_BLOCK_ENV, "")
    profile_path = os.environ.get(_PATH_ENV, "")
    if block_id not in _BLOCK_IDS or not profile_path:
        return
    global _ACTIVE_STATE, _BATCH_COUNTER
    try:
        _BATCH_COUNTER += 1
        request_id = f"{block_id}:unresolved-scheduler-batch:{_BATCH_COUNTER}"
        requests = getattr(batch, "reqs", None)
        if requests is not None and len(requests) == 1:
            observed = getattr(requests[0], "rid", None)
            if isinstance(observed, str) and observed:
                request_id = observed
        _ACTIVE_STATE = _State(
            runner,
            block_id,
            profile_path,
            _BATCH_COUNTER,
            request_id,
        )
        _bind_runner(runner)
    except Exception:
        _ACTIVE_STATE = None


def observe_host_merge(layer: int, operation: Callable[[], Any]) -> Any:
    """Observe the Host merge after the existing stream wait without adding sync."""
    state = _active_for_layer(layer)
    if state is None:
        return operation()
    parent_span_id, correlation_id = _parent_context(state)
    return _host_call(
        state,
        component="HOST_MERGE",
        layer=layer,
        operation=operation,
        args=(),
        kwargs={},
        parent_span_id=parent_span_id,
        correlation_id=correlation_id,
    )


def observe_npu_call(
    boundary: str,
    layer: int,
    operation: Callable[..., Any],
    *args: Any,
    **kwargs: Any,
) -> Any:
    """Record an NPU event pair and defer querying it until scheduler-after."""
    state = _active_for_layer(layer, device_sample=True)
    if state is None:
        return operation(*args, **kwargs)
    if boundary not in ("NPU_SHARED", "NPU_RESIDENT"):
        return operation(*args, **kwargs)
    try:
        import torch

        start = torch.npu.Event(enable_timing=True)
        end = torch.npu.Event(enable_timing=True)
        stream_id = str(torch.npu.current_stream())
        parent_span_id, correlation_id = _parent_context(state)
        identity = state.identity(
            boundary,
            layer,
            parent_span_id=parent_span_id,
            correlation_id=correlation_id,
            stream_id=stream_id,
        )
        start.record()
    except Exception:
        return operation(*args, **kwargs)
    outcome = "error"
    try:
        value = operation(*args, **kwargs)
        outcome = "return"
        return value
    finally:
        try:
            end.record()
            identity["outcome"] = outcome
            state.pending.append((start, end, identity))
        except Exception:
            pass


def _resolve_pending(state: _State) -> None:
    waiting: list[tuple[object, object, dict[str, Any]]] = []
    for start, end, identity in state.pending:
        try:
            if not end.query():
                waiting.append((start, end, identity))
                continue
            elapsed_ms = float(start.elapsed_time(end))
            if elapsed_ms < 0:
                raise ValueError("negative device event duration")
            state.rows.append(
                {
                    **identity,
                    "clock_domain": "NPU_EVENT_ELAPSED_MS",
                    "start_ns": None,
                    "end_ns": None,
                    "duration_ns": None,
                    "elapsed_ms": elapsed_ms,
                    "status": "COMPLETE",
                    "query_policy": "DEFERRED_QUERY_NO_SYNCHRONIZE",
                    "derivation": "DIRECT_OBSERVATION",
                }
            )
        except Exception:
            state.rows.append(
                {
                    **identity,
                    "clock_domain": "NPU_EVENT_ELAPSED_MS",
                    "start_ns": None,
                    "end_ns": None,
                    "duration_ns": None,
                    "elapsed_ms": None,
                    "status": "EVENT_QUERY_ERROR",
                    "query_policy": "DEFERRED_QUERY_NO_SYNCHRONIZE",
                }
            )
    state.pending = waiting


def _flush(state: _State) -> None:
    if not state.rows:
        return
    with open(state.profile_path, "a", encoding="utf-8") as destination:
        for row in state.rows:
            destination.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
    state.rows.clear()


def _flush_ready_states() -> None:
    waiting: list[_State] = []
    for state in _PENDING_STATES:
        _resolve_pending(state)
        _flush(state)
        if state.pending:
            waiting.append(state)
    _PENDING_STATES[:] = waiting


def after(runner: object) -> None:
    """Resolve ready events without synchronization and append JSONL rows."""
    if not _enabled():
        return
    global _ACTIVE_STATE
    state = _ACTIVE_STATE
    if state is None or state.runner is not runner:
        return
    try:
        _flush_ready_states()
        _resolve_pending(state)
        _flush(state)
        if state.pending:
            _PENDING_STATES.append(state)
    except Exception:
        pass
    finally:
        _ACTIVE_STATE = None


def _reset_for_tests() -> None:
    global _ACTIVE_STATE, _BATCH_COUNTER
    _ACTIVE_STATE = None
    _PENDING_STATES.clear()
    _BATCH_COUNTER = 0


__all__ = (
    "after",
    "before",
    "observe_host_merge",
    "observe_npu_call",
)
