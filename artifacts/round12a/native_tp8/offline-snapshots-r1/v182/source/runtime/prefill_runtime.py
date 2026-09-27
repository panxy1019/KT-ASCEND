"""Serial layerwise phase coordinator; optional real scheduler consumer.

One private transport phase spans all layers of a plain EXTEND batch. Request
telemetry spans all contiguous input chunks. Decode never enters the transport.
"""
from .prefill_telemetry import PrefillCollector, completion_event


class PrefillRuntime:
    def __init__(self, transport, *, threshold_tokens, layers=48, experts=512,
                 topk=10, max_tokens=32, event=None):
        if type(threshold_tokens) is not int or threshold_tokens < 1:
            raise ValueError('positive layerwise threshold required')
        if type(max_tokens) is not int or not 0 < max_tokens <= 32:
            raise ValueError('bounded layerwise token limit required')
        self.transport = transport
        self.threshold = threshold_tokens
        self.geometry = dict(layers=layers, experts=experts, topk=topk)
        self.max_tokens = max_tokens
        self.event = event or (lambda item: None)
        self.collector = None
        self.request = None
        self.identity = None
        self.epoch = None
        self.active = False
        self.state = 'READY'

    def _ready(self):
        if self.state != 'READY': raise RuntimeError(self.state)

    def _clear(self):
        self.request = self.collector = self.identity = None

    def _poison(self, primary):
        self.state = 'BROKEN_RELOAD_REQUIRED'
        cleanup = None
        if (self.active and getattr(self.transport, 'state', 'READY') == 'READY' and
                getattr(self.transport, '_active', True) is not None):
            try:
                self.transport.abort(self.identity[0], self.epoch)
                self.active = False
            except BaseException as error:
                cleanup = error
                fail_stop = getattr(self.transport, '_fail_stop', None)
                if fail_stop is not None: fail_stop(error)
        self._clear()  # Own collector is discarded; no partial statistics escape.
        if cleanup is not None: raise primary from cleanup
        raise primary

    def _retire_cancelled(self):
        if self.active or self.request is None: return
        reason = getattr(self.request, 'finished_reason', None)
        if reason is None or reason.__class__.__name__ != 'FINISH_ABORT': return
        rid, epoch, _ = self.identity
        self.collector.abort(rid)
        self._clear()
        try: self.event(dict(status='PREFILL_ABORT', request_id=rid, generation=epoch))
        except BaseException as error: self._poison(error)

    def before(self, batch, *, placement_epoch):
        self._ready()
        if self.active: raise RuntimeError('previous prefill batch has not drained')
        self._retire_cancelled()
        if not batch.forward_mode.is_extend(): return False
        if len(batch.reqs) != 1: raise ValueError('layerwise requires serial requests')
        req = batch.reqs[0]
        total, start, count = len(req.origin_input_ids), req.last_extend_prefix_len, req.extend_input_len
        if total < self.threshold: return False
        if type(count) is not int or not 0 < count <= self.max_tokens:
            raise ValueError('prefill chunk exceeds measured transport bound')
        identity = (req.rid, placement_epoch, total)
        if self.identity is not None and identity != self.identity:
            raise ValueError('prefill request/placement identity changed between chunks')
        if self.request is None:
            if start != 0: raise ValueError('prefix sharing is not admitted for layerwise')
            self.collector = PrefillCollector(**self.geometry)
            self.identity, self.epoch = identity, placement_epoch
        self.request = req
        self.collector.begin_batch(req.rid, placement_epoch, start=start, count=count, total=total)
        try:
            self.transport.begin(req.rid, placement_epoch)
            self.active = True
            self.event(dict(status='LAYERWISE_BATCH_BEGIN', request_id=req.rid,
                            generation=placement_epoch, start=start, count=count))
        except BaseException as error: self._poison(error)
        return True

    @staticmethod
    def _rows(value):
        return value.detach().cpu().tolist() if hasattr(value, 'detach') else value

    def apply(self, layer, hidden, ids, weights):
        self._ready()
        if not self.active: raise RuntimeError('unleased layerwise dispatch')
        try:
            self.collector.observe(layer, self._rows(ids), self._rows(weights))
            return self.transport.execute_layer(self.identity[0], self.epoch, layer, hidden, ids, weights)
        except BaseException as error: self._poison(error)

    def after(self):
        self._ready()
        self._retire_cancelled()
        if not self.active: return None
        req = self.request
        rid = self.identity[0]
        reason = getattr(req, 'finished_reason', None)
        aborted = reason is not None and reason.__class__.__name__ == 'FINISH_ABORT'
        try:
            if aborted: self.transport.abort(rid, self.epoch)
            else: self.transport.finish(rid, self.epoch)
            self.active = False
            complete = self.collector.end_batch(aborted=aborted)
            stats = None
            if complete:
                stats = dict(self.collector.stats.snapshot(), request_id=rid, generation=self.epoch)
            if complete or aborted: self._clear()
            event = (completion_event(stats, request_id=rid, generation=self.epoch)
                     if complete else
                     dict(status='PREFILL_ABORT' if aborted else 'LAYERWISE_BATCH_COMPLETE',
                          request_id=rid, generation=self.epoch))
            self.event(event)
            return stats
        except BaseException as error: self._poison(error)
