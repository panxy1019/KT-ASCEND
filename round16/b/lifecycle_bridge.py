"""Generation-safe request slots for asynchronous CPU expert work."""

from __future__ import annotations

from dataclasses import dataclass, field
from threading import Lock
from typing import Any

from round16.common.contracts import StateLease


@dataclass(frozen=True)
class RequestLease:
    state_lease: StateLease
    slot_id: int

    def __post_init__(self) -> None:
        if type(self.slot_id) is not int:
            raise ValueError("slot_id must be an integer")
        if self.slot_id < 0:
            raise ValueError("slot_id must be non-negative")

    @property
    def request_id(self) -> str:
        return self.state_lease.request_id

    @property
    def generation(self) -> int:
        return self.state_lease.allocation_generation

    @property
    def identity(self) -> tuple[Any, ...]:
        return (*self.state_lease.identity, self.slot_id)


@dataclass(frozen=True)
class TaskIdentity:
    lease: RequestLease
    sequence: int
    purpose: str

    def __post_init__(self) -> None:
        if type(self.purpose) is not str or not self.purpose:
            raise ValueError("task purpose must be a non-empty string")

    @property
    def identity(self) -> tuple[Any, ...]:
        return (*self.lease.identity, self.sequence, self.purpose)


@dataclass
class _SlotState:
    lease: RequestLease
    cancelled: bool = False
    next_sequence: int = 0
    pending: dict[tuple[Any, ...], TaskIdentity] = field(default_factory=dict)
    results: dict[int, Any] = field(default_factory=dict)


class RequestLeaseManager:
    def __init__(self, *, slot_count: int) -> None:
        if type(slot_count) is not int:
            raise ValueError("slot_count must be an integer")
        if slot_count <= 0:
            raise ValueError("slot_count must be positive")
        self._slot_count = slot_count
        self._lock = Lock()
        self._generations = [0] * slot_count
        self._active: dict[int, _SlotState] = {}

    def _require_state(self, lease: RequestLease) -> _SlotState:
        state = self._active.get(lease.slot_id)
        if state is None or state.lease != lease:
            raise RuntimeError("request lease is stale or no longer active")
        return state

    def acquire(
        self,
        request_id: str,
        *,
        request_epoch: int,
        allocation_generation: int | None = None,
        batch_generation: int,
        layer_id: int,
        token_range: tuple[int, int],
        completion_id: str,
        slot_id: int,
        writable_state: tuple[str, ...] = (),
    ) -> RequestLease:
        if type(request_id) is not str or not request_id:
            raise ValueError("request_id must be a non-empty string")
        if type(completion_id) is not str or not completion_id:
            raise ValueError("completion_id must be a non-empty string")
        if type(slot_id) is not int:
            raise ValueError("slot_id must be an integer")
        if slot_id < 0 or slot_id >= self._slot_count:
            raise ValueError("slot_id is out of range")
        with self._lock:
            if slot_id in self._active:
                raise RuntimeError("active request slot must drain before reuse")
            previous_generation = self._generations[slot_id]
            next_generation = (
                previous_generation + 1
                if allocation_generation is None
                else allocation_generation
            )
            if type(next_generation) is not int or next_generation <= previous_generation:
                raise ValueError(
                    "allocation_generation must be strictly greater than the slot generation"
                )
            state_lease = StateLease(
                request_id=request_id,
                request_epoch=request_epoch,
                allocation_generation=next_generation,
                batch_generation=batch_generation,
                layer_id=layer_id,
                token_range=token_range,
                completion_id=completion_id,
                writable_state=writable_state,
            )
            lease = RequestLease(state_lease, slot_id)
            self._generations[slot_id] = next_generation
            self._active[slot_id] = _SlotState(lease)
            return lease

    def assert_active(self, lease: RequestLease) -> None:
        """Fail before dispatch when a lease is stale or already cancelled."""

        with self._lock:
            state = self._require_state(lease)
            if state.cancelled:
                raise RuntimeError("cancelled request cannot dispatch work")

    def begin_task(self, lease: RequestLease, purpose: str) -> TaskIdentity:
        if type(purpose) is not str or not purpose:
            raise ValueError("task purpose must be a non-empty string")
        with self._lock:
            state = self._require_state(lease)
            if state.cancelled:
                raise RuntimeError("cancelled request cannot submit new tasks")
            state.next_sequence += 1
            task = TaskIdentity(lease, state.next_sequence, purpose)
            state.pending[task.identity] = task
            return task

    def cancel(self, lease: RequestLease) -> None:
        with self._lock:
            self._require_state(lease).cancelled = True

    def abort_task(self, task: TaskIdentity) -> bool:
        """Retire a reserved task without publishing a result."""

        with self._lock:
            state = self._active.get(task.lease.slot_id)
            if state is None or state.lease != task.lease:
                return False
            return state.pending.pop(task.identity, None) is not None

    def complete(self, task: TaskIdentity, result: Any) -> bool:
        with self._lock:
            state = self._active.get(task.lease.slot_id)
            if state is None or state.lease != task.lease:
                return False
            if state.pending.pop(task.identity, None) is None:
                return False
            if state.cancelled:
                return False
            state.results[task.sequence] = result
            return True

    def drain(self, lease: RequestLease) -> tuple[Any, ...]:
        with self._lock:
            state = self._require_state(lease)
            if state.pending:
                raise RuntimeError("request has pending tasks and cannot drain")
            self._active.pop(lease.slot_id)
            if state.cancelled:
                return ()
            return tuple(state.results[index] for index in sorted(state.results))


__all__ = ("RequestLease", "RequestLeaseManager", "TaskIdentity")
