"""Bounded serial routing aggregates; only complete request phases are published.

No tensor/backend imports. Producers supply original logical IDs in token order.
Counters occupy O(layers*experts), independent of prompt length. Device producers
must bound each event/chunk. A cancelled phase contributes no calibration data.
"""
import copy
import math


class RouteStats:
    def __init__(self, *, layers=48, experts=512, topk=10):
        if any(type(x) is not int or x < 1 for x in (layers, experts, topk)) or topk > experts:
            raise ValueError('invalid route geometry')
        self.layers, self.experts, self.topk = layers, experts, topk
        self.counts = [[0] * experts for _ in range(layers)]
        self.mass = [[0.0] * experts for _ in range(layers)]
        self.completed = 0
        self.tokens = 0
        self._active = None

    def observe(self, request_id, generation, phase, layer, token_start, ids, weights=None):
        if (not isinstance(request_id, str) or not request_id or
                type(generation) is not int or generation < 0 or
                phase not in ('prefill', 'decode') or type(layer) is not int or
                not 0 <= layer < self.layers or type(token_start) is not int or token_start < 0):
            raise ValueError('invalid route identity/span')
        if not isinstance(ids, (list, tuple)) or not 0 < len(ids) <= 4096:
            raise ValueError('empty/oversized event')
        if weights is not None and len(weights) != len(ids):
            raise ValueError('weight rows mismatch')
        # Validate the entire event before touching either aggregate or cursor.
        for n, row in enumerate(ids):
            if (len(row) != self.topk or any(type(i) is not int or not 0 <= i < self.experts for i in row)
                    or len(set(row)) != len(row)):
                raise ValueError('invalid/duplicate route ID')
            if weights is not None and (len(weights[n]) != self.topk or
                    any(type(w) not in (int, float) or not math.isfinite(w) or w < 0 for w in weights[n])):
                raise ValueError('invalid route weight')
        key = (request_id, generation, phase, weights is not None)
        if self._active is None:
            if token_start != 0: raise ValueError('phase must start at zero')
            self._active = dict(key=key, ends=[0] * self.layers,
                                counts=[[0] * self.experts for _ in range(self.layers)],
                                mass=[[0.0] * self.experts for _ in range(self.layers)])
        a = self._active
        if a['key'] != key or a['ends'][layer] != token_start:
            raise ValueError('stale request/generation or overlapping/noncontiguous route span')
        for n, row in enumerate(ids):
            for j, expert in enumerate(row):
                a['counts'][layer][expert] += 1
                if weights is not None: a['mass'][layer][expert] += weights[n][j]
        a['ends'][layer] += len(ids)

    def finish(self, request_id, phase, token_count):
        a = self._active
        if (a is None or a['key'][0] != request_id or a['key'][2] != phase or
                type(token_count) is not int or token_count < 1 or
                a['ends'] != [token_count] * self.layers):
            raise ValueError('incomplete layer/token coverage')
        for layer in range(self.layers):
            for expert in range(self.experts):
                self.counts[layer][expert] += a['counts'][layer][expert]
                self.mass[layer][expert] += a['mass'][layer][expert]
        self.tokens += token_count
        self.completed += 1
        self._active = None

    def abort(self, request_id):
        if self._active is None or self._active['key'][0] != request_id:
            raise ValueError('request mismatch')
        self._active = None

    def snapshot(self):
        return copy.deepcopy(dict(schema=1, layers=self.layers, experts=self.experts,
            topk=self.topk, completed_phases=self.completed, tokens=self.tokens,
            pending=self._active is not None, counts=self.counts, weight_mass=self.mass))


def frequency_mask(stats, capacity):
    if (stats.get('schema') != 1 or stats.get('pending') is not False or
            type(stats.get('completed_phases')) is not int or stats['completed_phases'] < 1):
        raise ValueError('complete route statistics required')
    layers, experts, topk, tokens = (stats.get(k) for k in ('layers', 'experts', 'topk', 'tokens'))
    if (any(type(x) is not int or x < 1 for x in (layers, experts, topk, tokens)) or
            topk > experts or type(capacity) is not int or not 0 < capacity < experts):
        raise ValueError('invalid geometry/capacity')
    counts = stats.get('counts')
    if not isinstance(counts, list) or len(counts) != layers:
        raise ValueError('incomplete layers')
    result = []
    for row in counts:
        if (not isinstance(row, list) or len(row) != experts or
                any(type(c) is not int or c < 0 for c in row) or sum(row) != tokens * topk):
            raise ValueError('invalid token/expert count conservation')
        result.append(sorted(sorted(range(experts), key=lambda i: (-row[i], i))[:capacity]))
    return result
