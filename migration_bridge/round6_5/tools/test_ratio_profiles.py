import unittest
from build_ratio_profiles import build


class Profiles(unittest.TestCase):
    c = dict(n_routed_experts=64, num_hidden_layers=27, first_k_dense_replace=1, moe_layer_freq=1)

    def test_partition_and_budget(self):
        for r in (8, 16, 32, 64):
            p = build(self.c, list(range(1,27)), r)
            self.assertEqual(p['total_npu_experts'], 26*r)
            for row in p['rows']:
                self.assertEqual(len(row['npu_ids']),r)
                self.assertFalse(set(row['npu_ids']) & set(row['cpu_ids']))
                self.assertEqual(set(row['npu_ids']) | set(row['cpu_ids']),set(range(64)))

    def test_nested(self):
        sets=[set(build(self.c,[17],r)['rows'][16]['npu_ids']) for r in (8,16,32,64)]
        self.assertTrue(all(a < b for a,b in zip(sets,sets[1:])))

    def test_unchanged_other_layers(self):
        p=build(self.c,[17],32)
        self.assertEqual(p['total_npu_experts'],26*64-32)

    def test_bad_layers(self):
        for layers in ([0],[17,17],[27]):
            with self.assertRaises(ValueError): build(self.c,layers,32)

    def test_bad_budget(self):
        for r in (0,4,31,128):
            with self.assertRaises(ValueError): build(self.c,[17],r)

    def test_repeat(self):
        self.assertEqual(build(self.c,[17],16),build(self.c,[17],16))


if __name__=='__main__': unittest.main()
