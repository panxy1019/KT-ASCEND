"""Serial prefill span contract and bounded completion telemetry."""
import math

from .route_stats import RouteStats


def completion_event(statistics, *, request_id, generation):
    """Build one bounded, source-identified event for a complete prefill phase."""
    if (not isinstance(request_id, str) or not request_id or
            type(generation) is not int or generation < 0 or
            not isinstance(statistics, dict) or statistics.get('schema') != 1 or
            statistics.get('pending') is not False or
            statistics.get('completed_phases') != 1):
        raise ValueError('complete prefill telemetry identity required')
    layers, experts, topk, tokens = (
        statistics.get(name) for name in ('layers', 'experts', 'topk', 'tokens'))
    if (any(type(value) is not int or value < 1
            for value in (layers, experts, topk, tokens)) or topk > experts):
        raise ValueError('invalid prefill telemetry geometry')
    counts, mass = statistics.get('counts'), statistics.get('weight_mass')
    if (not isinstance(counts, list) or not isinstance(mass, list) or
            len(counts) != layers or len(mass) != layers):
        raise ValueError('incomplete prefill telemetry layers')
    for count_row, mass_row in zip(counts, mass):
        if (not isinstance(count_row, list) or len(count_row) != experts or
                any(type(value) is not int or value < 0 for value in count_row) or
                sum(count_row) != tokens * topk or
                not isinstance(mass_row, list) or len(mass_row) != experts or
                any(type(value) not in (int, float) or not math.isfinite(value) or value < 0
                    for value in mass_row)):
            raise ValueError('invalid prefill telemetry count or mass')
    return dict(status='PREFILL_COMPLETE', schema=1, request_id=request_id,
                generation=generation, token_range=[0, tokens], layers=layers,
                experts=experts, topk=topk, completed_phases=1,
                logical_expert_counts=[list(row) for row in counts],
                logical_expert_weight_mass=[list(row) for row in mass])


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
