import copy
import unittest

from migration_bridge.round9.runtime.dynamic import PlacementController
from migration_bridge.round9.runtime.route_stats import RouteStats


class Adapter:
    def __init__(self): self.operations = []; self.fail = None; self.syncs = 0
    def synchronize(self):
        self.syncs += 1
        if self.fail == ('sync', self.syncs): raise RuntimeError('sync failure')
        self.operations.append(('sync',))
    def copy_experts(self, layer, moves):
        if self.fail == ('copy', layer): raise RuntimeError('copy failure')
        self.operations.append(('copy', layer, moves))
    def publish(self, layer, slots, generation):
        if self.fail == ('publish', layer): raise RuntimeError('publication failure')
        self.operations.append(('publish', layer, slots, generation))


class PrefillDynamicTests(unittest.TestCase):
    def fixture(self):
        adapter = Adapter()
        c = PlacementController({0: (0, 1), 1: (0, 1)}, adapter, experts=4, resident=2)
        generation = c.begin('live')
        stats = RouteStats(layers=2, experts=4, topk=2)
        for layer in (0, 1):
            stats.observe('live', generation, 'prefill', layer, 0, [[2, 3]])
            c.observe('live', generation, layer, [2, 3])
        stats.finish('live', 'prefill', 1)
        return c, adapter, dict(stats.snapshot(), request_id='live', generation=generation)

    def migrate(self, c, stats, **kwargs):
        return c.commit_prefill('live', 0, stats, placement_consumers_drained=True, **kwargs)

    def test_live_lease_survives_and_decode_uses_new_epoch(self):
        c, adapter, stats = self.fixture()
        self.assertEqual(self.migrate(c, stats), 1)
        with self.assertRaises(RuntimeError): c.begin('other')
        with self.assertRaises(RuntimeError): c.plan()
        with self.assertRaises(RuntimeError): c.observe('live', 0, 0, [0])
        c.observe('live', 1, 0, [0])
        c.finish('live', consumers_drained=True)
        self.assertEqual(c.commit(c.plan()), 2)
        first_publish = next(i for i, e in enumerate(adapter.operations) if e[0] == 'publish')
        self.assertEqual(sum(e[0] == 'copy' for e in adapter.operations[:first_publish]), 2)

    def test_incomplete_changed_or_non_drained_stats_do_not_write(self):
        c, adapter, stats = self.fixture()
        for field, value in [('request_id', 'other'), ('generation', 7), ('layers', 1), ('pending', True), ('completed_phases', 2),
                             ('tokens', 2), ('counts', [[0, 0, 1, 1], [1, 0, 1, 0]])]:
            bad = copy.deepcopy(stats); bad[field] = value
            with self.assertRaises(ValueError): self.migrate(c, bad)
        with self.assertRaises(RuntimeError): c.commit_prefill('live', 0, stats, placement_consumers_drained=False)
        with self.assertRaises(RuntimeError): c.commit_prefill('live', 1, stats, placement_consumers_drained=True)
        self.assertEqual(adapter.operations, [])
        self.assertEqual(self.migrate(c, stats), 1)
        with self.assertRaises(RuntimeError): c.commit_prefill('live', 1, stats, placement_consumers_drained=True)

    def test_cancel_before_barrier_rejects_late_commit(self):
        c, adapter, stats = self.fixture()
        c.finish('live', consumers_drained=True)
        with self.assertRaises(RuntimeError): self.migrate(c, stats)
        self.assertEqual(adapter.operations, [])

    def test_noop_barrier_still_seals_once_without_releasing_lease(self):
        adapter = Adapter()
        c = PlacementController({0: (0, 1)}, adapter, experts=4, resident=2)
        c.begin('live'); c.observe('live', 0, 0, [0, 1])
        s = RouteStats(layers=1, experts=4, topk=2)
        s.observe('live', 0, 'prefill', 0, 0, [[0, 1]]); s.finish('live', 'prefill', 1)
        doc = dict(s.snapshot(), request_id='live', generation=0)
        self.assertEqual(self.migrate(c, doc), 0)
        with self.assertRaises(RuntimeError): self.migrate(c, doc)
        with self.assertRaises(RuntimeError): c.begin('other')
        self.assertEqual(adapter.operations, [])
        c.observe('live', 0, 0, [2]); c.finish('live', consumers_drained=True)
        self.assertEqual(c.commit(c.plan()), 1)

    def test_copy_sync_and_publish_failures_poison_live_model(self):
        for failure in [('copy', 0), ('copy', 1), ('sync', 1), ('sync', 2), ('sync', 3), ('publish', 1)]:
            with self.subTest(failure=failure):
                c, adapter, stats = self.fixture(); adapter.fail = failure
                with self.assertRaises(RuntimeError): self.migrate(c, stats)
                self.assertEqual(c.state, 'BROKEN_RELOAD_REQUIRED')
                with self.assertRaises(RuntimeError): c.observe('live', c.generation, 0, [0])
                with self.assertRaises(RuntimeError): c.finish('live', consumers_drained=True)


if __name__ == '__main__': unittest.main()
