import copy
import importlib
import unittest

from migration_bridge.round9.runtime.route_stats import RouteStats


class FrequencyManifestTests(unittest.TestCase):
    def api(self):
        return importlib.import_module('migration_bridge.round9.runtime.frequency')

    def fixture(self):
        stats = RouteStats()
        for layer in range(48):
            stats.observe('dev-1', 0, 'prefill', layer, 0, [list(range(502, 512))])
        stats.finish('dev-1', 'prefill', 1)
        identity = {key: str(i) * 64 for i, key in enumerate(
            ('model_manifest_sha256', 'tokenizer_sha256', 'runtime_sha256',
             'corpus_sha256', 'placement_sha256'), 1)}
        return stats.snapshot(), identity

    def test_repeatable_per_layer_capacity_and_identity(self):
        api = self.api(); stats, identity = self.fixture()
        a = api.build_frequency(stats, identity, capacity=64)
        self.assertEqual(a, api.build_frequency(stats, identity, capacity=64))
        self.assertEqual(api.validate_frequency(a, identity, capacity=64), a['residents'])
        self.assertTrue(all(len(row) == 64 and row[-10:] == list(range(502, 512))
                            for row in a['residents']))
        b = api.build_frequency(stats, identity, capacity=32)
        self.assertTrue(all(len(row) == 32 for row in b['residents']))
        self.assertNotEqual(a['payload_sha256'], b['payload_sha256'])

    def test_no_fallback_for_missing_stale_or_corrupted_stats(self):
        api = self.api(); stats, identity = self.fixture()
        a = api.build_frequency(stats, identity, capacity=64)
        for mutation in ('mask', 'identity', 'stats', 'capacity'):
            bad = copy.deepcopy(a)
            if mutation == 'mask': bad['residents'][0][0] = 500
            elif mutation == 'identity': bad['identity']['model_manifest_sha256'] = 'a' * 64
            elif mutation == 'stats': bad['statistics']['counts'][0][511] = 10
            else: bad['capacity'] = 32
            with self.assertRaises(ValueError): api.validate_frequency(bad, identity, capacity=64)
        for key in identity:
            bad = dict(identity); del bad[key]
            with self.assertRaises(ValueError): api.build_frequency(stats, bad, capacity=64)
        bad = copy.deepcopy(stats); bad['layers'] = 47
        with self.assertRaises(ValueError): api.build_frequency(bad, identity, capacity=64)

    def test_digest_cannot_legitimize_wrong_deterministic_mask(self):
        api = self.api(); stats, identity = self.fixture()
        a = api.build_frequency(stats, identity, capacity=64)
        a['residents'][0][0] = 450
        a['residents'][0].sort()
        a['payload_sha256'] = api.payload_digest({k: v for k, v in a.items() if k != 'payload_sha256'})
        with self.assertRaises(ValueError): api.validate_frequency(a, identity, capacity=64)


if __name__ == '__main__': unittest.main()
