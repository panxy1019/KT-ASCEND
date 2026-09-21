"""Serial prefill span contract, shared by calibration and future prefill updates."""
from .route_stats import RouteStats


class PrefillCollector:
    def __init__(self, *, layers=48, experts=512, topk=10):
        self.stats = RouteStats(layers=layers, experts=experts, topk=topk)
        self.request = None
        self.batch = None
        self.cursor = 0

    def begin_batch(self, request_id, generation, *, start, count, total):
        if (not isinstance(request_id, str) or not request_id or
                type(generation) is not int or generation < 0 or
                any(type(n) is not int for n in (start, count, total)) or
                start < 0 or not 0 < count <= 4096 or start + count > total):
            raise ValueError('invalid prefill batch identity/span')
        key = (request_id, generation, total)
        if self.batch is not None or start != self.cursor or (self.request is not None and self.request != key):
            raise ValueError('active batch or stale request/placement epoch/span')
        self.request = key
        self.batch = dict(start=start, count=count, layers=set())

    def observe(self, layer, ids, weights=None):
        b = self.batch
        if b is None or layer in b['layers'] or len(ids) != b['count']:
            raise ValueError('unleased, duplicate or incomplete layer batch')
        self.stats.observe(self.request[0], self.request[1], 'prefill', layer,
                           b['start'], ids, weights)
        b['layers'].add(layer)

    def end_batch(self, *, aborted=False):
        if self.batch is None: raise ValueError('no active prefill batch')
        if aborted:
            if self.stats.snapshot()['pending']: self.stats.abort(self.request[0])
            self.batch = self.request = None
            self.cursor = 0
            return False
        if self.batch['layers'] != set(range(self.stats.layers)):
            raise ValueError('incomplete prefill layer coverage')
        self.cursor += self.batch['count']
        self.batch = None
        if self.cursor != self.request[2]: return False
        self.stats.finish(self.request[0], 'prefill', self.cursor)
        self.request = None
        self.cursor = 0
        return True

    def abort(self, request_id):
        if self.request is None or self.request[0] != request_id:
            raise ValueError('cancelled request identity mismatch')
        if self.stats.snapshot()['pending']: self.stats.abort(request_id)
        self.batch = self.request = None
        self.cursor = 0
