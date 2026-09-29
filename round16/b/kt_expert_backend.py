"""CPU-only submission adapter for KTransformers expert work."""

from __future__ import annotations

from dataclasses import dataclass
from threading import Lock
from typing import Any, Callable, Mapping, Sequence

from .routing_bridge import HybridRoutePlan


KT_MOE_WRAPPER_FORWARD_API = (
    "kt-kernel@e75740b3+d208f1d7:submit-sync-weighted-partial:v1"
)
KT_STREAM_HANDLE_API = "round16-b:kt-stream-handle-provider:v1"


@dataclass(frozen=True)
class CPUExpertTask:
    identity_key: tuple[Any, ...]
    logical_expert_id: int
    route_keys: tuple[tuple[int, int, int], ...]
    hidden_rows: tuple[Any, ...]


class KTExpertBackend:
    """Chunk CPU routes without reading or taking ownership of expert banks."""

    def __init__(
        self,
        worker: Callable[[CPUExpertTask], Sequence[Sequence[float]] | Mapping[tuple[int, int, int], Sequence[float]]],
        *,
        chunk_size: int = 32,
    ) -> None:
        if not callable(worker):
            raise TypeError("worker must be callable")
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        self._worker = worker
        self._chunk_size = chunk_size
        self._lock = Lock()
        self._submission_generation = 0

    def execute(
        self,
        plan: HybridRoutePlan,
        hidden_states: Sequence[Any],
        *,
        identity_key: tuple[Any, ...],
    ) -> dict[tuple[int, int, int], tuple[float, ...]]:
        if len(hidden_states) != plan.routes.token_count:
            raise ValueError("hidden state rows must match route tokens")
        if not plan.cpu_routes:
            return {}
        with self._lock:
            self._submission_generation += 1
            submission = self._submission_generation

        by_expert: dict[int, list[Any]] = {}
        for route in plan.cpu_routes:
            if route.cpu_logical_expert_id is None:
                raise ValueError("CPU route is missing its logical expert id")
            by_expert.setdefault(route.cpu_logical_expert_id, []).append(route)

        outputs: dict[tuple[int, int, int], tuple[float, ...]] = {}
        for expert_id, routes in by_expert.items():
            for chunk_index, offset in enumerate(range(0, len(routes), self._chunk_size)):
                chunk = routes[offset:offset + self._chunk_size]
                route_keys = tuple(route.key for route in chunk)
                task = CPUExpertTask(
                    (*identity_key, "cpu-moe", submission, expert_id, chunk_index),
                    expert_id,
                    route_keys,
                    tuple(hidden_states[route.token_index] for route in chunk),
                )
                result = self._worker(task)
                if isinstance(result, Mapping):
                    if set(result) != set(route_keys):
                        raise ValueError("worker mapping does not match submitted route keys")
                    rows = tuple(result[key] for key in route_keys)
                else:
                    rows = tuple(result)
                    if len(rows) != len(route_keys):
                        raise ValueError("worker returned the wrong number of route outputs")
                for key, row in zip(route_keys, rows):
                    outputs[key] = tuple(float(value) for value in row)
        return outputs


@dataclass(frozen=True)
class KTWeightedPartial:
    """KT output after Top-K weights, before model routed scaling."""

    output: tuple[tuple[float, ...], ...]
    route_weights_applied: bool = True
    routed_scaling_applied: bool = False

    def __post_init__(self) -> None:
        if self.route_weights_applied is not True:
            raise ValueError("KT partial must already include original Top-K weights")
        if self.routed_scaling_applied is not False:
            raise ValueError("KT partial must not include routed scaling")


@dataclass(frozen=True)
class KTPendingWeightedPartial:
    """One submitted KT operation, bound to a decode/work identity."""

    sequence: int
    identity_key: tuple[Any, ...]
    token_count: int
    hidden_states: Any
    stream_handle: Any


class KTWeightedPartialBackend:
    """Dependency-injected wrapper for the frozen KT submit/sync ABI.

    The wrapper itself stays external and is never imported here.  Its result is
    a token-level CPU routed partial that already includes the original router
    weights.  The caller remains the sole owner of routed scaling.
    """

    def __init__(
        self,
        wrapper: Any,
        *,
        stream_handle_provider: Callable[[tuple[Any, ...]], Any],
        api_identity: str,
    ) -> None:
        if api_identity != KT_MOE_WRAPPER_FORWARD_API:
            raise ValueError("KT MoE wrapper submit/sync API identity mismatch")
        if not callable(getattr(wrapper, "submit_forward", None)) or not callable(
            getattr(wrapper, "sync_forward", None)
        ):
            raise TypeError("KT wrapper must expose submit_forward and sync_forward")
        if not callable(stream_handle_provider) or getattr(
            stream_handle_provider, "round16_api_identity", None
        ) != KT_STREAM_HANDLE_API:
            raise ValueError("KT stream handle provider API identity mismatch")
        self._wrapper = wrapper
        self._stream_handle_provider = stream_handle_provider
        self._lock = Lock()
        self._next_sequence = 0
        self._pending: dict[int, KTPendingWeightedPartial] = {}
        self._submission_reserved = False
        self._poisoned = False

    def submit(
        self,
        plan: HybridRoutePlan,
        hidden_states: Sequence[Any],
        *,
        identity_key: tuple[Any, ...],
    ) -> KTPendingWeightedPartial:
        if len(hidden_states) != plan.routes.token_count:
            raise ValueError("hidden state rows must match route tokens")
        if not plan.cpu_routes:
            raise ValueError("KT weighted-partial execute requires a CPU route")
        with self._lock:
            if self._poisoned:
                raise RuntimeError(
                    "KT native submission state is unresolved after a submit error"
                )
            if self._submission_reserved or self._pending:
                raise RuntimeError(
                    "KT native ABI without work IDs permits only one outstanding task"
                )
            self._submission_reserved = True
        try:
            stream_handle = self._stream_handle_provider(identity_key)
            # The actual KT wrapper owns the Q32 mask and receives the original
            # logical Top-K routes.  Passing remapped or reweighted routes here
            # would change CPU ownership and risks applying weights twice.
            try:
                self._wrapper.submit_forward(
                    hidden_states,
                    plan.routes.topk_ids,
                    plan.routes.topk_weights,
                    stream_handle,
                )
            except BaseException:
                # A wrapper exception cannot prove whether native FIFO work was
                # enqueued.  Keep this backend permanently fail-closed rather
                # than risk pairing a later sync with the wrong decode.
                with self._lock:
                    self._poisoned = True
                raise
        except BaseException:
            with self._lock:
                self._submission_reserved = False
            raise
        with self._lock:
            self._next_sequence += 1
            pending = KTPendingWeightedPartial(
                self._next_sequence,
                identity_key,
                plan.routes.token_count,
                hidden_states,
                stream_handle,
            )
            self._pending[pending.sequence] = pending
            self._submission_reserved = False
        return pending

    def sync(self, pending: KTPendingWeightedPartial) -> KTWeightedPartial:
        if not isinstance(pending, KTPendingWeightedPartial):
            raise TypeError("KT sync requires a pending weighted-partial handle")
        with self._lock:
            if self._pending.get(pending.sequence) != pending:
                raise RuntimeError("KT pending handle is stale or already synchronized")
        result = self._wrapper.sync_forward(
            pending.hidden_states, pending.stream_handle
        )
        # Native returned, so its single FIFO item is drained even if output
        # validation below rejects the payload.
        with self._lock:
            self._pending.pop(pending.sequence, None)
        try:
            rows = tuple(tuple(float(value) for value in row) for row in result)
        except (TypeError, ValueError) as error:
            raise ValueError("KT sync result must be finite numeric token rows") from error
        if len(rows) != pending.token_count:
            raise ValueError("KT sync result row count must match route tokens")
        widths = {len(row) for row in rows}
        if len(widths) > 1:
            raise ValueError("KT sync result rows must have equal width")
        from math import isfinite

        if any(not isfinite(value) for row in rows for value in row):
            raise ValueError("KT sync result must be finite numeric token rows")
        return KTWeightedPartial(rows)

    def drain_discard(self, pending: KTPendingWeightedPartial) -> bool:
        """Synchronize already-submitted work while deliberately dropping output."""

        with self._lock:
            active = self._pending.get(pending.sequence) == pending
        if not active:
            return False
        self.sync(pending)
        return True

    def execute(
        self,
        plan: HybridRoutePlan,
        hidden_states: Sequence[Any],
        *,
        identity_key: tuple[Any, ...],
    ) -> KTWeightedPartial:
        """Synchronous convenience API; the Hybrid runner uses submit/sync."""

        return self.sync(
            self.submit(plan, hidden_states, identity_key=identity_key)
        )


__all__ = (
    "CPUExpertTask",
    "KTExpertBackend",
    "KT_MOE_WRAPPER_FORWARD_API",
    "KT_STREAM_HANDLE_API",
    "KTPendingWeightedPartial",
    "KTWeightedPartial",
    "KTWeightedPartialBackend",
)
