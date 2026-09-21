"""Serial exact-input PLE consumer bridge; no device allocation at import time.

One scheduler batch owns one payload. Gathering starts at PLE.prepare, while
the event wait is deferred until the PLE layer consumes the embedding. N-gram
history remains owned and committed by the original model forward.
"""
from .framework_config import BufferIdentity
from .ple_async import PREFILL_INPUT, SAMPLED_DECODE


def _values(tensor):
    return tensor.detach().cpu().tolist()


class DeferredPLERows:
    def __init__(self, owner, stamp, shape, *, ticket=None, rows=None):
        self.owner, self.stamp = owner, stamp
        self.ticket, self.rows, self.shape = ticket, rows, tuple(shape)
        self.flattened = False

    def flatten(self, start_dim):
        if start_dim != 1 or self.flattened:
            raise ValueError('PLE embedding must flatten exactly once from dimension1')
        self.flattened = True
        return self


class PLEConsumerRuntime:
    def __init__(self, ring):
        self.ring = ring
        self.state = 'READY'
        self.context = None
        self.stamp = None
        self.prepared = self.consumed = False
        self.payload = None
        self.async_batches = self.sync_fallbacks = 0

    def _ready(self):
        if self.state != 'READY': raise RuntimeError(self.state)

    def begin(self, request_id, generation, input_ids, state_slot, *, phase):
        self._ready()
        BufferIdentity(request_id, generation, 0)
        if self.context is not None: raise RuntimeError('prior PLE batch has not drained')
        if (phase not in (PREFILL_INPUT, SAMPLED_DECODE) or
                not isinstance(input_ids, (list, tuple)) or not 0 < len(input_ids) <= 32 or
                any(type(v) is not int or v < 0 for v in input_ids) or
                type(state_slot) is not int or state_slot <= 0):
            raise ValueError('PLE requires bounded known input IDs and live state slot')
        self.context = (request_id, generation, tuple(input_ids), state_slot, phase)
        self.stamp = object()
        self.prepared = self.consumed = False
        self.payload = None

    def prepare(self, input_ids, state_slot, row_ids, sync_table):
        self._ready()
        if self.context is None or self.prepared:
            raise RuntimeError('unleased or duplicate PLE prepare')
        rid, epoch, tokens, slot, phase = self.context
        if tuple(_values(input_ids)) != tokens or _values(state_slot) != [slot]:
            raise ValueError('PLE input/state slot differs from scheduler batch identity')
        rows = _values(row_ids)
        if (len(rows) != len(tokens) or not rows or not rows[0] or
                any(not isinstance(row, list) or len(row) != len(rows[0]) for row in rows)):
            raise ValueError('PLE row shape differs from exact input batch')
        ticket = None
        try:
            expected = (len(rows), len(rows[0]), self.ring.row_width)
            from . import working_set_timeline as timeline
            import time
            submitted_ns = time.monotonic_ns() if timeline.enabled() else None
            ticket = self.ring.submit_exact(rid, epoch, rows, phase=phase)
            if submitted_ns is not None:
                timeline.event(
                    'ple', 'gather_submit',
                    duration_ns=time.monotonic_ns()-submitted_ns,
                    phase_name=phase, rows=len(rows)*len(rows[0]),
                    bytes=len(rows)*len(rows[0])*self.ring.row_width*2,
                    asynchronous=ticket is not None)
            if ticket is None:
                gather_ns = time.monotonic_ns() if timeline.enabled() else None
                gathered = sync_table.gather_rows(row_ids)
                if gather_ns is not None:
                    timeline.event(
                        'ple', 'sync_gather',
                        duration_ns=time.monotonic_ns()-gather_ns,
                        phase_name=phase, rows=len(rows)*len(rows[0]))
                if tuple(gathered.shape) != expected:
                    raise RuntimeError('sync PLE row shape differs from exact rows')
                self.payload = DeferredPLERows(
                    self, self.stamp, expected, rows=gathered,
                )
                self.sync_fallbacks += 1
            else:
                self.payload = DeferredPLERows(
                    self, self.stamp, expected, ticket=ticket,
                )
                self.async_batches += 1
            self.prepared = True
            return self.payload
        except BaseException as primary:
            self.state = 'BROKEN_RELOAD_REQUIRED'
            if ticket is not None:
                try: self.ring.cancel_and_drain(ticket)
                except BaseException as cleanup: raise primary from cleanup
            raise

    def consume(self, embeddings, consumer):
        self._ready()
        if self.context is None or not self.prepared or self.consumed:
            raise RuntimeError('unleased or duplicate PLE consumer')
        asynchronous = self.payload.ticket is not None
        primary = None
        cleanup_error = None
        acquired = False
        try:
            if (not isinstance(embeddings, DeferredPLERows)
                    or embeddings is not self.payload
                    or embeddings.owner is not self
                    or embeddings.stamp is not self.stamp
                    or not embeddings.flattened):
                raise ValueError('stale or unflattened PLE payload')
            if asynchronous:
                from . import working_set_timeline as timeline
                import time
                wait_ns = time.monotonic_ns() if timeline.enabled() else None
                embeddings = self.ring.wait(self.payload.ticket)
                if wait_ns is not None:
                    timeline.event(
                        'ple', 'gather_wait_and_h2d_dependency',
                        clock_domain='HOST_OBSERVED_DEVICE_OPERATION',
                        duration_ns=time.monotonic_ns()-wait_ns,
                        slot=getattr(self.payload.ticket, 'slot', None),
                        rows=getattr(self.payload.ticket, 'row_count', self.payload.shape[0]),
                        bytes=(getattr(self.payload.ticket, 'row_count', self.payload.shape[0])
                               * self.ring.row_width * 2))
                acquired = True
                if tuple(embeddings.shape) != self.payload.shape:
                    raise RuntimeError('async PLE device shape differs from exact rows')
            else:
                embeddings = self.payload.rows
            embeddings = embeddings.flatten(1)
            from . import working_set_timeline as timeline
            import time
            consumer_ns = time.monotonic_ns() if timeline.enabled() else None
            result = consumer(embeddings)
            if consumer_ns is not None:
                timeline.event(
                    'ple', 'consumer_enqueue',
                    clock_domain='HOST_OBSERVED_DEVICE_OPERATION',
                    duration_ns=time.monotonic_ns()-consumer_ns,
                    asynchronous=asynchronous)
        except BaseException as error:
            self.state = 'BROKEN_RELOAD_REQUIRED'
            primary = error
        if primary is not None and asynchronous and not acquired:
            try: self.ring.cancel_and_drain(self.payload.ticket)
            except BaseException as cleanup: cleanup_error = cleanup
        if acquired:
            try: self.ring.release(self.payload.ticket)
            except BaseException as cleanup:
                self.state = 'BROKEN_RELOAD_REQUIRED'
                if primary is not None: raise primary from cleanup
                raise
        if primary is not None:
            if cleanup_error is not None: raise primary from cleanup_error
            raise primary
        self.consumed = True
        return result

    def end(self):
        self._ready()
        if self.context is None: return
        if not self.prepared or not self.consumed:
            self.state = 'BROKEN_RELOAD_REQUIRED'
            primary = RuntimeError('PLE consumer did not drain before scheduler boundary')
            if (isinstance(self.payload, DeferredPLERows)
                    and self.payload.ticket is not None):
                try: self.ring.cancel_and_drain(self.payload.ticket)
                except BaseException as cleanup: raise primary from cleanup
            raise primary
        self.context = self.payload = self.stamp = None
        self.prepared = self.consumed = False


def bind_ple_consumer(ple, runtime):
    """Keep the original PLE arithmetic/state write as the sole consumer."""
    if getattr(ple, '_round9_async_ple', None) is not None:
        raise RuntimeError('PLE consumer already bound')
    original = ple.run
    def run(hidden_states, embeddings, state_slot, forward_batch):
        return runtime.consume(embeddings, lambda rows:
            original(
                hidden_states=hidden_states, embeddings=rows,
                state_slot=state_slot, forward_batch=forward_batch,
            ))
    ple._round9_async_ple = runtime
    ple.run = run
