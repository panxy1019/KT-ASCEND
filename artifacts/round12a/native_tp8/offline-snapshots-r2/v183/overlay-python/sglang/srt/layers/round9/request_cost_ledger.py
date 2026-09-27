"""Bounded host-clock accounting for opt-in request observation.

Inclusive and self time partition only a serial, properly nested host clock.
No device time, H2D bandwidth, TTFT or worker execution time is inferred.
"""
from contextlib import contextmanager
import time

STAGES = frozenset(('request', 'prefill_batch', 'decode_batch', 'expert_layer',
                    'route_collect', 'route_group', 'route_host_read', 'expert_group', 'prepare',
                    'compute_enqueue', 'device_wait', 'release', 'assemble',
                    'model_forward', 'result_processing'))


class CostLedger:
    def __init__(self, identity, *, max_events=20000, clock=time.monotonic_ns):
        if not isinstance(identity, dict) or not all(identity.get(k) for k in
                ('request_id', 'runtime_sha256', 'profile_sha256')):
            raise ValueError('request and build identity required')
        if type(max_events) is not int or max_events < 1:
            raise ValueError('positive event bound required')
        self.identity = dict(identity)
        self.max_events, self.clock = max_events, clock
        self.events, self.stack, self.errors = [], [], []
        self.attempted = self.dropped = 0
        self.closed = False

    @contextmanager
    def span(self, stage, *, layer=None, chunk=None, group=None,
             logical_bytes=0, expert_count=0):
        if self.closed or stage not in STAGES:
            raise ValueError('closed ledger or unknown stage')
        if any(type(v) is not int or v < 0 for v in (logical_bytes, expert_count)):
            raise ValueError('invalid count/bytes')
        start = self.clock()
        self.attempted += 1
        row = dict(id=self.attempted, parent=self.stack[-1]['id'] if self.stack else None,
                   stage=stage, layer=layer, chunk=chunk, group=group,
                   logical_bytes=logical_bytes, expert_count=expert_count,
                   start_ns=start, child_ns=0, outcome='RETURN')
        self.stack.append(row)
        try:
            yield row
        except BaseException:
            row['outcome'] = 'ERROR'
            raise
        finally:
            end = self.clock()
            if not self.stack or self.stack[-1] is not row:
                self.errors.append('NON_SERIAL_NESTING')
            else:
                self.stack.pop()
                duration = end - start
                if duration < row['child_ns'] or duration < 0:
                    self.errors.append('INVALID_HOST_CLOCK')
                row.update(end_ns=end, inclusive_ns=duration,
                           self_ns=duration-row.pop('child_ns'))
                if self.stack:
                    self.stack[-1]['child_ns'] += duration
                if len(self.events) < self.max_events:
                    self.events.append(row)
                else:
                    self.dropped += 1

    def finish(self):
        if self.closed:
            raise ValueError('duplicate finish')
        self.closed = True
        errors = list(self.errors)
        if self.stack:
            errors.append('UNFINISHED_SPANS')
        if self.dropped:
            errors.append('EVENT_LIMIT_EXCEEDED')
        roots = [r for r in self.events if r['parent'] is None]
        if len(roots) != 1 or roots[0]['stage'] != 'request':
            errors.append('REQUEST_ROOT_REQUIRED')
        if any(r['outcome'] != 'RETURN' for r in self.events):
            errors.append('PROVIDER_ERROR')
        complete = not errors
        totals = {}
        if complete:
            for row in self.events:
                entry = totals.setdefault(row['stage'], dict(calls=0, inclusive_ns=0,
                    self_ns=0, logical_bytes=0, expert_count=0))
                entry['calls'] += 1
                for key in ('inclusive_ns', 'self_ns', 'logical_bytes', 'expert_count'):
                    entry[key] += row[key]
        return dict(schema=1, identity=self.identity, clock='HOST_MONOTONIC_NS',
                    status='COMPLETE_HOST_ACCOUNTING' if complete else 'INVALID_EVIDENCE',
                    errors=errors, attempted=self.attempted, dropped=self.dropped,
                    events=self.events, totals=totals,
                    device_cost='UNKNOWN', bandwidth='UNKNOWN',
                    scope='HOST_SERIAL_ONLY_NOT_DEVICE_OR_FULL_REQUEST_QUALIFICATION')
