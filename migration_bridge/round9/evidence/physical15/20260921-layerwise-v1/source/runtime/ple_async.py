"""Default-off, bounded asynchronous transport for exact original PLE rows.

This module does not predict decode tokens and does not replace the synchronous
PLE reader.  Admission failure is immediate so a correctness caller can use the
synchronous reference path instead of waiting for optional work to drain.
"""
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import importlib
import threading

from .framework_config import BufferIdentity
from .host_budget import HostWorkBudget, Priority


PREFILL_INPUT = "prefill_input"
SAMPLED_DECODE = "sampled_decode"
_EXACT_PHASES = frozenset((PREFILL_INPUT, SAMPLED_DECODE))
_BF16_BYTES = 2
_SLOT_COUNT = 2


@dataclass(frozen=True, slots=True)
class PLEAsyncTicket:
    """Opaque lease for one exact request/placement/allocation generation."""

    identity: BufferIdentity
    slot: int
    row_shape: tuple
    row_count: int


@dataclass(slots=True)
class _SlotState:
    storage: object
    ticket: object = None
    allocation: object = None
    priority: object = None
    future: object = None
    device_rows: object = None
    event: object = None
    consumer_stream: object = None
    state: str = "IDLE"
    cancelled: bool = False


def _positive_int(value, name):
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _normalize_rows(rows, total_rows, max_rows):
    try:
        outer = list(rows)
    except TypeError as exc:
        raise ValueError("row_ids must be a nonempty rectangular sequence") from exc
    if not outer:
        raise ValueError("row_ids must be nonempty")

    nested = isinstance(outer[0], (list, tuple))
    if nested:
        width = len(outer[0])
        if width == 0 or any(not isinstance(row, (list, tuple)) or len(row) != width
                             for row in outer):
            raise ValueError("row_ids must be rectangular")
        flat = [row_id for row in outer for row_id in row]
        shape = (len(outer), width)
    else:
        if any(isinstance(row, (list, tuple)) for row in outer):
            raise ValueError("row_ids must be rectangular")
        flat = outer
        shape = (len(outer),)
    if len(flat) > max_rows:
        raise ValueError("row_ids exceed the bounded slot capacity")
    if any(type(row_id) is not int or not 0 <= row_id < total_rows for row_id in flat):
        raise ValueError("PLE row outside original table")
    return tuple(flat), shape


class AsyncPLERing:
    """Two reusable pinned slots for exact PLE gathers and nonblocking H2D.

    ``submit_exact`` returns ``None`` when disabled, resource credits are not
    immediately available, or both slots remain busy.  That result explicitly
    selects the existing synchronous reference path; this class never waits for
    capacity and therefore cannot deadlock required PLE traffic behind optional
    prefetch.
    """

    def __init__(self, reader, *, enabled=False, max_rows=1, budget=None, backend=None):
        if type(enabled) is not bool:
            raise ValueError("enabled must be boolean")
        _positive_int(max_rows, "max_rows")
        width = getattr(reader, "width", None)
        total_rows = getattr(reader, "total_rows", None)
        _positive_int(width, "reader.width")
        _positive_int(total_rows, "reader.total_rows")
        self.reader = reader
        self.enabled = enabled
        self.max_rows = max_rows
        self.row_width = width
        self.total_rows = total_rows
        self.budget = budget
        self.backend = backend
        self._slots = []
        self._lock = threading.RLock()
        self._state = "OFF" if not enabled else "READY"

        if not enabled:
            return
        if not isinstance(budget, HostWorkBudget):
            raise TypeError("enabled async PLE requires HostWorkBudget")
        if backend is None:
            backend = TorchNPUAsyncPLEBackend()
            self.backend = backend
        required = (
            "allocate_slots", "submit_gather", "launch_h2d", "query_event",
            "wait_event", "record_event", "synchronize_event", "release_slot", "close",
        )
        for name in required:
            if not callable(getattr(backend, name, None)):
                raise TypeError(f"backend is missing {name}()")
        storage = list(backend.allocate_slots(_SLOT_COUNT, max_rows, width))
        if len(storage) != _SLOT_COUNT:
            raise RuntimeError("backend must allocate exactly two PLE slots")
        self._slots = [_SlotState(item) for item in storage]

    def submit_exact(self, request_id, placement_epoch, row_ids, *, phase,
                     priority=Priority.PLE_CURRENT):
        """Submit known row IDs, or return ``None`` for synchronous fallback."""
        if not self.enabled:
            return None
        if phase not in _EXACT_PHASES:
            raise ValueError("phase must contain exact prefill or sampled decode IDs")
        if type(priority) is not Priority:
            raise ValueError("priority must be a Priority")
        # BufferIdentity performs the exact request and placement validation.
        BufferIdentity(request_id, placement_epoch, 0)
        flat, shape = _normalize_rows(row_ids, self.total_rows, self.max_rows)
        with self._lock:
            self._require_open()
            slot = self._idle_slot()
            if slot is None and priority is Priority.PLE_CURRENT:
                for candidate in tuple(self._slots):
                    if (candidate.priority is Priority.PLE_PREFETCH
                            and candidate.state != "LEASED"):
                        self._cancel_slot(candidate, wait=False)
                slot = self._idle_slot()
            if slot is None:
                return None

            byte_count = len(flat) * self.row_width * _BF16_BYTES
            allocation = self.budget.try_acquire(
                request_id=request_id,
                placement_epoch=placement_epoch,
                priority=priority,
                host_bytes=byte_count,
                h2d_bytes=byte_count,
                pinned_bytes=byte_count,
            )
            if allocation is None:
                return None
            ticket = PLEAsyncTicket(
                allocation.identity, self._slots.index(slot), shape, len(flat)
            )
            slot.ticket = ticket
            slot.allocation = allocation
            slot.priority = priority
            slot.cancelled = False
            slot.state = "GATHERING"
            try:
                self.budget.mark_in_flight(allocation)
                slot.future = self.backend.submit_gather(
                    self.reader, flat, slot.storage, ticket.identity
                )
                if not callable(getattr(slot.future, "done", None)) or not callable(
                        getattr(slot.future, "result", None)):
                    raise TypeError("backend submit_gather must return a Future-like object")
            except BaseException:
                self._clear_slot(slot, in_flight=True)
                raise
            return ticket

    def poll(self, ticket):
        """Advance one ticket without blocking and return its lifecycle state."""
        with self._lock:
            self._require_open()
            slot = self._ticket_slot(ticket)
            if slot.state == "GATHERING":
                if not slot.future.done():
                    return "CANCELLED_PENDING" if slot.cancelled else "GATHERING"
                try:
                    self._finish_gather(slot)
                except BaseException:
                    self._retire_after_failure(slot)
                    raise
                if slot.cancelled:
                    self._retire(slot, wait_event=False)
                    return "RETIRED"
                self._launch(slot)
            if slot.state == "TRANSFERRING" and self.backend.query_event(
                    slot.event, slot.ticket.identity):
                if slot.cancelled:
                    self._retire(slot, wait_event=False)
                    return "RETIRED"
                slot.state = "READY"
            elif (slot.cancelled and slot.state in ("READY", "LEASED")
                  and self.backend.query_event(slot.event, slot.ticket.identity)):
                self._retire(slot, wait_event=False)
                return "RETIRED"
            return slot.state

    def wait(self, ticket, consumer_stream=None):
        """Wait for CPU gather, enqueue H2D, and order a consumer stream after it."""
        with self._lock:
            self._require_open()
            slot = self._ticket_slot(ticket)
            if slot.cancelled:
                raise RuntimeError("cancelled PLE ticket cannot be consumed")
            if slot.state == "GATHERING":
                try:
                    self._finish_gather(slot)
                    self._launch(slot)
                except BaseException:
                    self._retire_after_failure(slot)
                    raise
            if slot.state not in ("TRANSFERRING", "READY"):
                raise RuntimeError(f"PLE ticket is not consumable in {slot.state}")
            slot.consumer_stream = self.backend.wait_event(
                slot.event, consumer_stream, ticket.identity
            )
            slot.state = "LEASED"
            return slot.device_rows

    def release(self, ticket):
        """Retire after work using the rows has been queued on the consumer stream.

        ``wait`` establishes the consumer stream.  The caller must enqueue its
        PLE use before calling ``release``; release records a new event there and
        waits for it before either pinned or device storage can be reused.
        """
        with self._lock:
            self._require_open()
            slot = self._ticket_slot(ticket)
            if slot.state not in ("TRANSFERRING", "READY", "LEASED"):
                raise RuntimeError("PLE ticket has no launched transfer to release")
            self._retire(slot, wait_event=True)

    def cancel(self, ticket):
        """Cancel ownership while retaining buffers through all in-flight work."""
        with self._lock:
            self._require_open()
            self._cancel_slot(self._ticket_slot(ticket), wait=True)

    def close(self):
        with self._lock:
            if self._state == "CLOSED":
                return
            if self.enabled:
                for slot in self._slots:
                    if slot.state == "IDLE":
                        continue
                    if not slot.cancelled:
                        self.budget.cancel(slot.allocation)
                        slot.cancelled = True
                    if slot.state == "GATHERING":
                        try:
                            self._finish_gather(slot)
                        except BaseException:
                            self._retire_after_failure(slot)
                            continue
                    self._retire(slot, wait_event=slot.event is not None)
                self.backend.close()
            self._state = "CLOSED"

    def snapshot(self):
        with self._lock:
            return {
                "state": self._state,
                "slots": len(self._slots),
                "active": sum(slot.state != "IDLE" for slot in self._slots),
                "cancelled_pending": sum(
                    slot.state != "IDLE" and slot.cancelled for slot in self._slots
                ),
            }

    def _require_open(self):
        if self._state == "CLOSED":
            raise RuntimeError("async PLE ring is closed")

    def _idle_slot(self):
        return next((slot for slot in self._slots if slot.state == "IDLE"), None)

    def _ticket_slot(self, ticket):
        if not isinstance(ticket, PLEAsyncTicket) or not 0 <= ticket.slot < len(self._slots):
            raise ValueError("invalid PLE ticket")
        slot = self._slots[ticket.slot]
        if slot.ticket is not ticket:
            raise ValueError("stale or mismatched PLE ticket")
        return slot

    def _finish_gather(self, slot):
        completed_identity = slot.future.result()
        if completed_identity != slot.ticket.identity:
            raise RuntimeError("PLE gather completed with mismatched identity")

    def _launch(self, slot):
        slot.device_rows, slot.event = self.backend.launch_h2d(
            slot.storage, slot.ticket.row_count, slot.ticket.row_shape,
            slot.ticket.identity,
        )
        slot.state = "TRANSFERRING"

    def _cancel_slot(self, slot, *, wait):
        if slot.cancelled:
            return
        self.budget.cancel(slot.allocation)
        slot.cancelled = True
        if slot.state == "GATHERING":
            if slot.future.done():
                try:
                    self._finish_gather(slot)
                except BaseException:
                    self._retire_after_failure(slot)
                    return
                self._retire(slot, wait_event=False)
            return
        if wait:
            self._retire(slot, wait_event=slot.event is not None)
        elif self.backend.query_event(slot.event, slot.ticket.identity):
            self._retire(slot, wait_event=False)

    def _retire(self, slot, *, wait_event):
        if wait_event:
            event = slot.event
            if slot.state == "LEASED":
                event = self.backend.record_event(
                    slot.consumer_stream, slot.ticket.identity
                )
            self.backend.synchronize_event(event, slot.ticket.identity)
        self.budget.complete(slot.allocation)
        self.budget.release(slot.allocation)
        self.backend.release_slot(slot.storage, slot.ticket.identity)
        self._reset(slot)

    def _retire_after_failure(self, slot):
        # A failed CPU gather has no device work. The real backend drains its
        # private stream before surfacing any launch failure that lacks an event.
        self.budget.complete(slot.allocation)
        self.budget.release(slot.allocation)
        self.backend.release_slot(slot.storage, slot.ticket.identity)
        self._reset(slot)

    def _clear_slot(self, slot, *, in_flight):
        if in_flight:
            self.budget.complete(slot.allocation)
        self.budget.release(slot.allocation)
        if slot.ticket is not None:
            self.backend.release_slot(slot.storage, slot.ticket.identity)
        self._reset(slot)

    @staticmethod
    def _reset(slot):
        slot.ticket = None
        slot.allocation = None
        slot.priority = None
        slot.future = None
        slot.device_rows = None
        slot.event = None
        slot.consumer_stream = None
        slot.state = "IDLE"
        slot.cancelled = False


@dataclass(slots=True)
class _TorchNPUSlot:
    host: object
    device: object
    host_bytes: object
    identity: object = None


@dataclass(frozen=True, slots=True)
class _TorchNPUEvent:
    event: object
    identity: BufferIdentity


class TorchNPUAsyncPLEBackend:
    """Real pinned-memory/side-stream backend, imported only when enabled."""

    def __init__(self, *, device="npu", workers=2):
        _positive_int(workers, "workers")
        self.torch = importlib.import_module("torch")
        self.torch_npu = importlib.import_module("torch_npu")
        npu = getattr(self.torch, "npu", None)
        if npu is None:
            raise RuntimeError("torch_npu did not install torch.npu")
        self.device = self.torch.device(device)
        self.stream = npu.Stream(device=self.device)
        self.executor = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="ple-row")

    def allocate_slots(self, count, max_rows, row_width):
        torch = self.torch
        slots = []
        for _ in range(count):
            host = torch.empty(
                (max_rows, row_width), dtype=torch.bfloat16,
                device="cpu", pin_memory=True,
            )
            # torch.frombuffer may be redirected to NPU by transfer_to_npu.
            # A byte view of the already-pinned CPU tensor keeps row gathering
            # on the host and avoids an intermediate unpinned tensor.
            host_bytes = memoryview(host.view(torch.uint8).numpy()).cast("B")
            slots.append(_TorchNPUSlot(
                host=host,
                device=torch.empty(
                    (max_rows, row_width), dtype=torch.bfloat16, device=self.device,
                ),
                host_bytes=host_bytes,
            ))
        return slots

    def submit_gather(self, reader, row_ids, slot, identity):
        if slot.identity is not None:
            raise RuntimeError("pinned PLE slot is already owned")
        slot.identity = identity

        def gather():
            raw = reader.gather_bytes(row_ids)
            expected = len(row_ids) * reader.width * _BF16_BYTES
            if len(raw) != expected:
                raise ValueError("PLE gather returned the wrong byte count")
            slot.host_bytes[:expected] = raw
            return identity

        return self.executor.submit(gather)

    def launch_h2d(self, slot, row_count, row_shape, identity):
        self._require(slot, identity)
        torch = self.torch
        try:
            with torch.npu.stream(self.stream):
                event = torch.npu.Event()
                slot.device[:row_count].copy_(slot.host[:row_count], non_blocking=True)
                event.record(self.stream)
        except BaseException:
            # If enqueue succeeded but event creation/recording failed, drain
            # the private stream before the ring returns this storage to use.
            self.stream.synchronize()
            raise
        shape = (*row_shape, slot.device.shape[1])
        return slot.device[:row_count].reshape(shape), _TorchNPUEvent(event, identity)

    def query_event(self, event, identity):
        return bool(self._event(event, identity).query())

    def wait_event(self, event, consumer_stream, identity):
        event = self._event(event, identity)
        stream = self.torch.npu.current_stream() if consumer_stream is None else consumer_stream
        stream.wait_event(event)
        return stream

    def record_event(self, consumer_stream, identity):
        return _TorchNPUEvent(consumer_stream.record_event(), identity)

    def synchronize_event(self, event, identity):
        self._event(event, identity).synchronize()

    def release_slot(self, slot, identity):
        self._require(slot, identity)
        slot.identity = None

    def close(self):
        self.executor.shutdown(wait=True, cancel_futures=False)

    @staticmethod
    def _require(slot, identity):
        if slot.identity != identity:
            raise RuntimeError("stale or mismatched pinned PLE slot identity")

    @staticmethod
    def _event(event, identity):
        if not isinstance(event, _TorchNPUEvent) or event.identity != identity:
            raise RuntimeError("stale or mismatched PLE event identity")
        return event.event


__all__ = (
    "AsyncPLERing", "PLEAsyncTicket", "TorchNPUAsyncPLEBackend",
    "PREFILL_INPUT", "SAMPLED_DECODE", "Priority", "BufferIdentity",
)
