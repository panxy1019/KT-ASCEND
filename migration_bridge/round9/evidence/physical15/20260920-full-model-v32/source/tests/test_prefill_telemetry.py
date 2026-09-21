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


if __name__ == '__main__': unittest.main()
