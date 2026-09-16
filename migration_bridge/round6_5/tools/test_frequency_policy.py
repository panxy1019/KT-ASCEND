import unittest
from frequency_policy import resident_ids,project


class Frequency(unittest.TestCase):
    def test_tie_rule(self):self.assertEqual(resident_ids([1]*64,8),list(range(8)))
    def test_frequency_first(self):
        counts=[0]*64;counts[63]=2
        self.assertEqual(resident_ids(counts,8),list(range(7))+[63])
    def test_complete_model_denominator(self):
        rows={l:dict(ids=list(range(6)),weights=[1/6]*6) for l in range(1,27)}
        masks={l:set(range(3)) for l in rows}
        r=project(rows,masks);self.assertEqual(r['cpu_events'],78);self.assertTrue(r['touches_cpu'])
        self.assertAlmostEqual(r['cpu_weight_mass'],13)
    def test_missing_layer_rejects(self):
        with self.assertRaises(ValueError):project({}, {})


if __name__=='__main__':unittest.main()
