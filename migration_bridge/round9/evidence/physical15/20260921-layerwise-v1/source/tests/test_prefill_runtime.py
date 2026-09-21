import unittest
from types import SimpleNamespace as NS

from migration_bridge.round9.runtime.prefill_runtime import PrefillRuntime


class Transport:
    def __init__(self): self.events = []; self._active = None; self.layers = set(); self.state = 'READY'
    def begin(self, rid, epoch): self.events.append(('begin', rid, epoch)); self._active = rid; self.layers = set()
    def execute_layer(self, rid, epoch, layer, hidden, ids, weights):
        self.events.append(('layer', layer)); self.layers.add(layer); return hidden
    def finish(self, rid, epoch):
        if self.layers != {0, 1}: raise ValueError('incomplete transport')
        self.events.append(('finish', rid, epoch)); self._active = None
    def abort(self, rid, epoch): self.events.append(('abort', rid, epoch)); self._active = None


def batch(req, *, extend=True):
    return NS(reqs=[req], forward_mode=NS(is_extend=lambda: extend))


class PrefillRuntimeTests(unittest.TestCase):
    def request(self, total=4, count=2, start=0):
        return NS(rid='r', origin_input_ids=list(range(total)),
                  extend_input_len=count, last_extend_prefix_len=start,
                  finished_reason=None)

    def forward(self, runtime):
        for layer in range(2):
            hidden = [[1.0], [2.0]]
            ids, weights = [[0, 1], [1, 2]], [[.5, .5], [.25, .75]]
            self.assertEqual(runtime.apply(layer, hidden, ids, weights), hidden)

    def test_two_chunks_complete_real_layer_spans_and_telemetry(self):
        transport = Transport(); events = []
        runtime = PrefillRuntime(transport, threshold_tokens=3, layers=2,
                                 experts=4, topk=2, event=events.append)
        req = self.request()
        self.assertTrue(runtime.before(batch(req), placement_epoch=0))
        self.forward(runtime)
        self.assertIsNone(runtime.after())
        req.last_extend_prefix_len = 2
        runtime.before(batch(req), placement_epoch=0); self.forward(runtime)
        stats = runtime.after()
        self.assertEqual(stats['request_id'], 'r')
        self.assertEqual(stats['generation'], 0)
        self.assertEqual(stats['completed_phases'], 1)
        self.assertEqual(len([x for x in transport.events if x[0] == 'finish']), 2)
        self.assertEqual(events[-1]['status'], 'PREFILL_COMPLETE')
        self.assertFalse(runtime.before(batch(req, extend=False), placement_epoch=0))

    def test_short_hybrid_and_decode_do_not_engage_transport(self):
        transport = Transport()
        runtime = PrefillRuntime(transport, threshold_tokens=8, layers=2, experts=4, topk=2)
        req = self.request()
        self.assertFalse(runtime.before(batch(req), placement_epoch=0))
        self.assertIsNone(runtime.after())
        self.assertFalse(runtime.before(batch(req, extend=False), placement_epoch=0))
        self.assertEqual(transport.events, [])

    def test_abort_discards_partial_stats_and_allows_fresh_request(self):
        transport = Transport()
        runtime = PrefillRuntime(transport, threshold_tokens=1, layers=2, experts=4, topk=2)
        req = self.request(); runtime.before(batch(req), placement_epoch=0); self.forward(runtime)
        req.finished_reason = type('FINISH_ABORT', (), {})()
        self.assertIsNone(runtime.after())
        self.assertEqual(transport.events[-1][0], 'abort')
        req = self.request(); req.rid = 'new'
        self.assertTrue(runtime.before(batch(req), placement_epoch=0))

    def test_stale_epoch_incomplete_layers_and_oversize_fail_closed(self):
        for change in ('epoch', 'layers', 'oversize'):
            runtime = PrefillRuntime(Transport(), threshold_tokens=1, layers=2, experts=4, topk=2)
            req = self.request()
            if change == 'oversize':
                req.origin_input_ids = list(range(40)); req.extend_input_len = 33
                with self.assertRaises(ValueError): runtime.before(batch(req), placement_epoch=0)
                continue
            runtime.before(batch(req), placement_epoch=0)
            if change == 'layers':
                with self.assertRaises(ValueError): runtime.after()
                self.assertEqual(runtime.state, 'BROKEN_RELOAD_REQUIRED')
            else:
                self.forward(runtime); runtime.after(); req.last_extend_prefix_len = 2
                with self.assertRaises(ValueError): runtime.before(batch(req), placement_epoch=1)

    def test_failure_retires_real_phase_and_telemetry(self):
        for stage in ('duplicate', 'incomplete', 'event_begin', 'event_end'):
            transport = Transport()
            def event(item):
                if stage == 'event_begin' or (stage == 'event_end' and item['status'] == 'PREFILL_COMPLETE'):
                    raise OSError('evidence sink failed')
            runtime = PrefillRuntime(transport, threshold_tokens=1, layers=2, experts=4, topk=2, event=event)
            req = self.request(total=2)
            with self.assertRaises((ValueError, OSError)):
                runtime.before(batch(req), placement_epoch=0)
                if stage == 'duplicate':
                    runtime.apply(0, [[1], [2]], [[0, 1], [1, 2]], [[.5, .5]] * 2)
                    runtime.apply(0, [[1], [2]], [[0, 1], [1, 2]], [[.5, .5]] * 2)
                if stage == 'event_end': self.forward(runtime)
                runtime.after()
            self.assertIsNone(transport._active)
            self.assertIsNone(runtime.collector)
            self.assertEqual(runtime.state, 'BROKEN_RELOAD_REQUIRED')

    def test_abort_between_chunks_retires_on_idle_or_decode(self):
        for idle in (True, False):
            transport = Transport(); events = []
            runtime = PrefillRuntime(transport, threshold_tokens=1, layers=2, experts=4, topk=2,event=events.append)
            req = self.request(); runtime.before(batch(req), placement_epoch=0)
            self.forward(runtime); runtime.after()
            req.finished_reason = type('FINISH_ABORT', (), {})()
            if idle: runtime.after()
            else: runtime.before(batch(req, extend=False), placement_epoch=0)
            self.assertIsNone(runtime.collector)
            self.assertEqual(events[-1]['status'], 'PREFILL_ABORT')
            self.assertFalse(any(e[0] == 'abort' for e in transport.events))

    def test_same_identity_wrapper_continuation_is_valid(self):
        runtime = PrefillRuntime(Transport(), threshold_tokens=1, layers=2, experts=4, topk=2)
        runtime.before(batch(self.request()), placement_epoch=0); self.forward(runtime); runtime.after()
        runtime.before(batch(self.request(start=2)), placement_epoch=0); self.forward(runtime)
        self.assertEqual(runtime.after()['tokens'], 4)


if __name__ == '__main__': unittest.main()
