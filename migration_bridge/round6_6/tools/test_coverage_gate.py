import copy
import unittest
from summarize_three_way import validate


class CoverageTest(unittest.TestCase):
    def fixture(self):
        return {'rows':[dict(layer=l,expert=e,call=c,qlen=11 if c==1 else 1,
            deterministic=True,finite=True,local_safety_pass=True,
            production_hashes=['a']*10,npu_hashes=['b']*10,
            comparisons={k:{'relative_l2':0.001} for k in ['production_vs_npu','production_vs_canonical_fp32']},relative_error_ratio=1.)
            for l in [1,9,17,26] for e in range(1,64,2) for c in [2,33,55,1]]}
    def test_complete_does_not_promote(self):
        self.assertFalse(validate(self.fixture())['promotion_allowed'])
    def test_duplicate_rejected(self):
        d=self.fixture();d['rows'][-1]=copy.deepcopy(d['rows'][0])
        with self.assertRaises(ValueError):validate(d)
    def test_nonfinite_rejected(self):
        d=self.fixture();d['rows'][0]['finite']=False
        with self.assertRaises(ValueError):validate(d)
    def test_fake_repeat_flag_rejected(self):
        d=self.fixture();d['rows'][0]['production_hashes'][-1]='changed'
        with self.assertRaises(ValueError):validate(d)
    def test_fake_safety_flag_rejected(self):
        d=self.fixture();d['rows'][0]['comparisons']['production_vs_npu']['relative_l2']=.2
        with self.assertRaises(ValueError):validate(d)


if __name__=='__main__':unittest.main()
