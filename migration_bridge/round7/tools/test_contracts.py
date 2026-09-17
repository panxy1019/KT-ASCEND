import unittest
from contracts import validate_mask,stable_frequency_order,validate_generation,quality_gate


class Contracts(unittest.TestCase):
    def test_nonuniform_budget_including_zero(self):
        counts=[0,16]+[8]*24
        rows=[dict(layer=l,npu_ids=list(range(n)),cpu_ids=list(range(n,64))) for l,n in enumerate(counts,1)]
        self.assertEqual(validate_mask(rows),208)
        rows[1]['npu_ids'].append(16)
        with self.assertRaises(ValueError): validate_mask(rows)

    def test_ties_and_zero_slots(self):
        self.assertEqual(stable_frequency_order([0]*64,8),list(range(8)))
        self.assertEqual(stable_frequency_order([0]*64,0),[])
        c=[0]*64;c[61]=9;c[32]=9
        self.assertEqual(stable_frequency_order(c,2),[32,61])

    def test_generation_mismatch(self):
        validate_generation(3,3,3,3)
        with self.assertRaises(ValueError):validate_generation(4,3,4,4)

    def test_exact_product_boundary(self):
        self.assertTrue(quality_gate(162,0))
        self.assertFalse(quality_gate(161,0))
        self.assertFalse(quality_gate(174,1))


if __name__=='__main__':unittest.main()
