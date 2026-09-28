"""Bounded per-request ownership for Round14a C1/C2/C4 batches.

This module deliberately contains no scheduling policy.  It only splits one
real SGLang ``ForwardBatch`` into request-owned token spans and multiplexes the
already-qualified serial QSA and PLE runtimes.  Read-only model weights remain
shared; every mutable request identity remains distinct.
"""

from __future__ import annotations

import os


ALLOWED_CONCURRENCY = (1, 2, 4)


def configured_max() -> int:
    raw = os.environ.get("ROUND14A_CONCURRENCY_MAX", "1")
    try:
        value = int(raw)
    except ValueError as error:
        raise RuntimeError("ROUND14A_CONCURRENCY_MAX must be an integer") from error
    if value not in ALLOWED_CONCURRENCY:
        raise RuntimeError("Round14a concurrency must be one of 1,2,4")
    return value


def enabled() -> bool:
    return configured_max() > 1


def _cpu_values(value):
    if hasattr(value, "detach"):
        value = value.detach().cpu().tolist()
    return list(value)


def request_slices(forward_batch, token_count: int):
    """Return ``(ordinal, req_slot, start, stop)`` for every active request.

    Decode owns exactly one row per request.  EXTEND rows are concatenated in
    the same order as ``extend_seq_lens_cpu``.  No inferred or padded length is
    accepted because that could cross request state.
    """
    batch_size = int(forward_batch.batch_size)
    maximum = configured_max()
    if not 1 <= batch_size <= maximum:
        raise RuntimeError(
            f"Round14a batch size {batch_size} exceeds bounded maximum {maximum}"
        )
    slots = [int(value) for value in _cpu_values(forward_batch.req_pool_indices)]
    if len(slots) != batch_size or len(set(slots)) != batch_size:
        raise RuntimeError("Round14a request pool slots must be unique and complete")
    mode = forward_batch.forward_mode
    if mode.is_decode():
        lengths = [1] * batch_size
    elif mode.is_extend():
        raw = getattr(forward_batch, "extend_seq_lens_cpu", None)
        if raw is None:
            raw = getattr(forward_batch, "extend_seq_lens", None)
        if raw is None:
            raise RuntimeError("Round14a EXTEND batch lacks exact per-request lengths")
        lengths = [int(value) for value in _cpu_values(raw)]
        if len(lengths) != batch_size or any(value <= 0 for value in lengths):
            raise RuntimeError("Round14a EXTEND request lengths are invalid")
    else:
        raise RuntimeError(f"Round14a rejects forward mode {mode!r}")
    if sum(lengths) != int(token_count):
        raise RuntimeError(
            f"Round14a request spans total {sum(lengths)}, expected {token_count}"
        )
    result = []
    start = 0
    for ordinal, (slot, length) in enumerate(zip(slots, lengths)):
        stop = start + length
        result.append((ordinal, slot, start, stop))
        start = stop
    return tuple(result)


def validate_scheduler_requests(batch, *, threshold: int = 129):
    """Bind scheduler request objects to their pool slots and H129 identity."""
    requests = tuple(batch.reqs)
    maximum = configured_max()
    if not 1 <= len(requests) <= maximum:
        raise RuntimeError("Round14a scheduler batch exceeds bounded concurrency")
    rows = []
    seen = set()
    for request in requests:
        slot = getattr(request, "req_pool_idx", None)
        if slot is None:
            raise RuntimeError("Round14a request lacks an allocated pool slot")
        slot = int(slot)
        if slot in seen:
            raise RuntimeError("Round14a scheduler request slots are not unique")
        seen.add(slot)
        total = len(request.origin_input_ids)
        if total >= threshold:
            raise RuntimeError(
                f"Round14a concurrent request {request.rid} total {total} violates H129"
            )
        rows.append((slot, request, total))
    return tuple(rows)


class BatchedContextReferenceRuntime:
    """One qualified serial QSA runtime per live request identity."""

    def __init__(self, factory, *, event=None):
        self._factory = factory
        self._event = event or (lambda item: None)
        self._owners = {}
        self._by_slot = {}
        self._active = ()

    @property
    def active(self):
        return bool(self._active)

    @property
    def next_layer(self):
        if not self._active:
            return 0
        values = {runtime.next_layer for unused_slot, unused_request, runtime in self._active}
        return values.pop() if len(values) == 1 else -1

    def begin_batch(self, batch):
        if self._active:
            raise RuntimeError("prior Round14a QSA batch did not drain")
        active = []
        for slot, request, total in validate_scheduler_requests(batch):
            runtime = self._owners.get(request)
            if runtime is None:
                runtime = self._factory()
                self._owners[request] = runtime
            elif runtime.request_token is not request:
                raise RuntimeError("Round14a QSA request identity changed")
            runtime.begin(request.rid, request)
            if slot in self._by_slot and self._by_slot[slot][0] is not request:
                raise RuntimeError("Round14a QSA pool slot was reused while live")
            self._by_slot[slot] = (request, runtime)
            active.append((slot, request, runtime))
            self._event({"status": "ROUND14A_CONCURRENCY_REQUEST_BOUND",
                         "plane": "QSA", "request_id": request.rid,
                         "req_pool_slot": slot, "request_total_tokens": total,
                         "batch_size": len(batch.reqs)})
        self._active = tuple(active)

    def consume_for(self, req_slot, *args, **kwargs):
        owner = self._by_slot.get(int(req_slot))
        if owner is None or not any(owner[1] is runtime for unused_slot,
                                    unused_request, runtime in self._active):
            raise RuntimeError("Round14a QSA consume used an unleased request slot")
        return owner[1].consume(*args, **kwargs)

    def after_batch(self):
        if not self._active:
            raise RuntimeError("Round14a QSA batch is unleased")
        active, self._active = self._active, ()
        for slot, request, runtime in active:
            finished = bool(request.finished())
            runtime.after(request, finished=finished)
            if finished:
                self._owners.pop(request, None)
                self._by_slot.pop(slot, None)

    def abort_request(self, request):
        runtime = self._owners.get(request)
        if runtime is None:
            return False
        if any(owner is request for unused_slot, owner, unused_runtime in self._active):
            raise RuntimeError("Round14a QSA abort reached an active device batch")
        runtime.abort(request)
        self._owners.pop(request, None)
        slot = getattr(request, "req_pool_idx", None)
        if slot is not None:
            self._by_slot.pop(int(slot), None)
        return True

    def close(self):
        if self._active or self._owners:
            raise RuntimeError("Round14a QSA runtimes remain live at close")


class BatchedPLEConsumerRuntime:
    """One serial PLE consumer per request, sharing the bounded read-only ring."""

    def __init__(self, ring, factory, *, event=None):
        self.ring = ring
        self._factory = factory
        self._event = event or (lambda item: None)
        self._owners = {}
        self._by_state_slot = {}
        self._active = ()
        self.async_batches = 0
        self.sync_fallbacks = 0

    @property
    def context(self):
        return None if not self._active else tuple(runtime.context for _, runtime in self._active)

    def begin_batch(self, batch, *, generation: int, phase: str):
        if self._active:
            raise RuntimeError("prior Round14a PLE batch did not drain")
        active = []
        for slot, request, total in validate_scheduler_requests(batch):
            state_slot = getattr(request, "mamba_pool_idx", None)
            if state_slot is None:
                raise RuntimeError("Round14a PLE request state is not allocated")
            state_slot = int(state_slot)
            if state_slot <= 0:
                raise RuntimeError("Round14a PLE request has padding state slot")
            runtime = self._owners.get(request)
            if runtime is None:
                runtime = self._factory(self.ring)
                self._owners[request] = runtime
            tokens = list(batch.input_ids.detach().cpu().tolist())
            # Scheduler input is concatenated.  Use the request's exact chunk.
            start = sum(int(req.extend_input_len) if batch.forward_mode.is_extend() else 1
                        for req in batch.reqs[:len(active)])
            count = int(request.extend_input_len) if batch.forward_mode.is_extend() else 1
            request_tokens = tokens[start:start + count]
            runtime.begin(request.rid, generation, request_tokens, state_slot, phase=phase)
            if (state_slot in self._by_state_slot
                    and self._by_state_slot[state_slot][0] is not request):
                raise RuntimeError("Round14a PLE state slot was reused while live")
            self._by_state_slot[state_slot] = (request, runtime)
            active.append((request, runtime))
            self._event({"status": "ROUND14A_CONCURRENCY_REQUEST_BOUND",
                         "plane": "PLE", "request_id": request.rid,
                         "req_pool_slot": slot, "state_slot": state_slot,
                         "request_total_tokens": total,
                         "batch_size": len(batch.reqs)})
        self._active = tuple(active)

    def prepare(self, input_ids, state_slot, row_ids, sync_table):
        values = state_slot.detach().cpu().tolist()
        if len(values) != 1:
            raise RuntimeError("Round14a PLE prepare must name one request state")
        owner = self._by_state_slot.get(int(values[0]))
        if owner is None or owner not in self._active:
            raise RuntimeError("Round14a PLE prepare used an unleased state slot")
        return owner[1].prepare(input_ids, state_slot, row_ids, sync_table)

    def consume(self, embeddings, consumer):
        runtime = getattr(embeddings, "owner", None)
        if runtime is None or not any(runtime is item for unused, item in self._active):
            raise RuntimeError("Round14a PLE consumer received a stale payload")
        return runtime.consume(embeddings, consumer)

    def after_batch(self):
        if not self._active:
            raise RuntimeError("Round14a PLE batch is unleased")
        active, self._active = self._active, ()
        rows = []
        for request, runtime in active:
            context = runtime.context
            asynchronous = runtime.async_batches > 0
            runtime.end()
            self.async_batches += runtime.async_batches
            self.sync_fallbacks += runtime.sync_fallbacks
            runtime.async_batches = runtime.sync_fallbacks = 0
            rows.append((request, context, asynchronous))
            if request.finished():
                self._owners.pop(request, None)
                self._by_state_slot.pop(int(context[3]), None)
        return tuple(rows)

    def abort_request(self, request):
        runtime = self._owners.get(request)
        if runtime is None:
            return False
        if any(owner is request for owner, unused in self._active):
            raise RuntimeError("Round14a PLE abort reached an active device batch")
        if runtime.context is not None:
            raise RuntimeError("Round14a PLE abort retained an undrained payload")
        self._owners.pop(request, None)
        state_slot = getattr(request, "mamba_pool_idx", None)
        if state_slot is not None:
            self._by_state_slot.pop(int(state_slot), None)
        return True

    def close(self):
        if self._active or self._owners:
            raise RuntimeError("Round14a PLE runtimes remain live at close")
        self.ring.close()


__all__ = (
    "ALLOWED_CONCURRENCY", "configured_max", "enabled", "request_slices",
    "validate_scheduler_requests", "BatchedContextReferenceRuntime",
    "BatchedPLEConsumerRuntime",
)
