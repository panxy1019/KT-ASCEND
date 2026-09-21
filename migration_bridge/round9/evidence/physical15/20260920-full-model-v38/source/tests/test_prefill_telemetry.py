import importlib
import unittest


class PrefillTelemetryTests(unittest.TestCase):
    def make(self):
        api = importlib.import_module('migration_bridge.round9.runtime.prefill_telemetry')
        return api.PrefillCollector(layers=2, experts=4, topk=2)

    def test_chunks_publish_only_complete_prompt(self):
        c = self.make()
        c.begin_batch('r', 0, start=0, count=2, total=3)
        for layer in (0, 1): c.observe(layer, [[0, 1], [2, 3]])
        self.assertFalse(c.end_batch())
        self.assertEqual(c.stats.snapshot()['completed_phases'], 0)
        c.begin_batch('r', 0, start=2, count=1, total=3)
        for layer in (0, 1): c.observe(layer, [[2, 3]])
        self.assertTrue(c.end_batch())
        self.assertEqual(c.stats.snapshot()['counts'], [[1, 1, 2, 2]] * 2)

    def test_abort_reclaims_pending_and_new_request_starts_clean(self):
        c = self.make()
        c.begin_batch('old', 0, start=0, count=1, total=2)
        for layer in (0, 1): c.observe(layer, [[0, 1]])
        self.assertFalse(c.end_batch(aborted=True))
        self.assertFalse(c.stats.snapshot()['pending'])
        c.begin_batch('new', 1, start=0, count=1, total=1)
        for layer in (0, 1): c.observe(layer, [[2, 3]])
        c.end_batch()
        self.assertEqual(c.stats.snapshot()['counts'], [[0, 0, 1, 1]] * 2)

    def test_missing_layer_stale_epoch_duplicate_and_bad_span_rejected(self):
        c = self.make()
        with self.assertRaises(ValueError): c.begin_batch('r', 0, start=1, count=1, total=2)
        c.begin_batch('r', 0, start=0, count=1, total=2)
        c.observe(0, [[0, 1]])
        with self.assertRaises(ValueError): c.observe(0, [[0, 1]])
        with self.assertRaises(ValueError): c.end_batch()
        c.observe(1, [[0, 1]]); c.end_batch()
        with self.assertRaises(ValueError): c.begin_batch('r', 1, start=1, count=1, total=2)
        with self.assertRaises(ValueError): c.begin_batch('r', 0, start=1, count=2, total=2)
        c.begin_batch('r', 0, start=1, count=1, total=2)
        with self.assertRaises(ValueError): c.observe(0, [[0, 1], [2, 3]])
        for layer in (0, 1): c.observe(layer, [[2, 3]])
        c.end_batch()

    def test_cancel_between_chunks_preserves_previous_complete_statistics(self):
        c = self.make()
        c.begin_batch('done', 0, start=0, count=1, total=1)
        for layer in (0, 1): c.observe(layer, [[2, 3]])
        c.end_batch()
        expected = c.stats.snapshot()
        c.begin_batch('abort', 0, start=0, count=1, total=2)
        for layer in (0, 1): c.observe(layer, [[0, 1]])
        c.end_batch()
        with self.assertRaises(ValueError): c.abort('wrong')
        c.abort('abort')
        self.assertEqual(expected, c.stats.snapshot())

    def test_completion_event_binds_identity_span_counts_and_mass(self):
        api = importlib.import_module('migration_bridge.round9.runtime.prefill_telemetry')
        c = self.make()
        c.begin_batch('r', 7, start=0, count=2, total=2)
        ids = [[0, 1], [1, 2]]
        weights = [[.5, .5], [.25, .75]]
        for layer in (0, 1): c.observe(layer, ids, weights)
        self.assertTrue(c.end_batch())
        stats = c.stats.snapshot()
        event = api.completion_event(stats, request_id='r', generation=7)
        self.assertEqual(event['status'], 'PREFILL_COMPLETE')
        self.assertEqual(event['request_id'], 'r')
        self.assertEqual(event['generation'], 7)
        self.assertEqual(event['token_range'], [0, 2])
        self.assertEqual(event['logical_expert_counts'], [[1, 2, 1, 0]] * 2)
        self.assertEqual(event['logical_expert_weight_mass'], [[.5, .75, .75, 0.0]] * 2)

    def test_completion_event_rejects_partial_or_malformed_statistics(self):
        api = importlib.import_module('migration_bridge.round9.runtime.prefill_telemetry')
        c = self.make()
        c.begin_batch('r', 0, start=0, count=1, total=1)
        for layer in (0, 1): c.observe(layer, [[0, 1]], [[.4, .6]])
        pending = c.stats.snapshot()
        with self.assertRaises(ValueError):
            api.completion_event(pending, request_id='r', generation=0)
        c.end_batch(); complete = c.stats.snapshot()
        for field, value in (('request_id', ''), ('generation', -1)):
            args = dict(request_id='r', generation=0); args[field] = value
            with self.assertRaises(ValueError): api.completion_event(complete, **args)
        malformed = dict(complete, counts=[[1, 1, 0, 0]])
        with self.assertRaises(ValueError):
            api.completion_event(malformed, request_id='r', generation=0)


if __name__ == '__main__': unittest.main()
