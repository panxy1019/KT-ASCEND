"""Reject ambiguous routing traces before creating replayable placement masks."""
import importlib
import unittest


class RouteStatsTests(unittest.TestCase):
    def api(self):
        spec = importlib.util.find_spec('migration_bridge.round9.runtime.route_stats')
        self.assertIsNotNone(spec, 'shared route statistics implementation required')
        return importlib.import_module(spec.name)

    def test_counts_mass_ties_and_layer_coverage(self):
        api = self.api()
        stats = api.RouteStats(layers=2, experts=4, topk=2)
        stats.observe('a', 0, 'prefill', 0, 0, [[3, 1], [3, 2]], [[.6, .4], [.7, .3]])
        stats.observe('a', 0, 'prefill', 1, 0, [[0, 2], [2, 1]], [[.5, .5], [.8, .2]])
        stats.finish('a', 'prefill', 2)
        doc = stats.snapshot()
        self.assertEqual(doc['counts'], [[0, 1, 1, 2], [1, 1, 2, 0]])
        self.assertAlmostEqual(doc['weight_mass'][0][3], 1.3)
        self.assertEqual(api.frequency_mask(doc, 2), [[1, 3], [0, 2]])
        self.assertEqual(stats.snapshot(), doc)

    def test_partial_invalid_observation_cannot_mutate_or_replay(self):
        api = self.api()
        stats = api.RouteStats(layers=2, experts=4, topk=2)
        with self.assertRaises(ValueError):
            stats.observe('a', 0, 'prefill', 0, 0, [[0, 1], [2, 4]])
        stats.observe('a', 0, 'prefill', 0, 0, [[0, 1]])
        for args in [('a', 0, 'prefill', 0, 0, [[0, 1]]),
                     ('a', 1, 'prefill', 1, 0, [[0, 1]]),
                     ('b', 0, 'prefill', 1, 0, [[0, 1]])]:
            with self.assertRaises(ValueError): stats.observe(*args)
        with self.assertRaises(ValueError): stats.finish('a', 'prefill', 1)
        with self.assertRaises(ValueError): api.frequency_mask(stats.snapshot(), 2)
        stats.observe('a', 0, 'prefill', 1, 0, [[2, 3]])
        stats.finish('a', 'prefill', 1)
        self.assertEqual(stats.snapshot()['counts'], [[1, 1, 0, 0], [0, 0, 1, 1]])

    def test_chunked_trace_has_identical_aggregate(self):
        api = self.api()
        a = api.RouteStats(layers=1, experts=4, topk=2)
        b = api.RouteStats(layers=1, experts=4, topk=2)
        a.observe('r', 7, 'prefill', 0, 0, [[1, 3], [0, 1]])
        b.observe('r', 7, 'prefill', 0, 0, [[1, 3]])
        b.observe('r', 7, 'prefill', 0, 1, [[0, 1]])
        for s in (a, b): s.finish('r', 'prefill', 2)
        self.assertEqual(a.snapshot(), b.snapshot())

    def test_abort_discards_partial_request_and_snapshot_is_detached(self):
        api = self.api()
        s = api.RouteStats(layers=1, experts=4, topk=2)
        s.observe('old', 2, 'prefill', 0, 0, [[0, 1]])
        s.abort('old')
        s.observe('new', 2, 'prefill', 0, 0, [[2, 3]])
        s.finish('new', 'prefill', 1)
        d = s.snapshot(); d['counts'][0][0] = 100
        self.assertEqual(api.frequency_mask(s.snapshot(), 2), [[2, 3]])

    def test_nonfinite_weights_duplicate_experts_and_empty_stats_rejected(self):
        api = self.api()
        s = api.RouteStats(layers=1, experts=4, topk=2)
        for ids, weights in [([[0, 0]], None), ([[0, 1]], [[float('nan'), .2]]),
                             ([[0, 1]], [[-.1, .2]])]:
            with self.assertRaises(ValueError): s.observe('r', 0, 'prefill', 0, 0, ids, weights)
        with self.assertRaises(ValueError): api.frequency_mask(s.snapshot(), 2)


if __name__ == '__main__': unittest.main()
