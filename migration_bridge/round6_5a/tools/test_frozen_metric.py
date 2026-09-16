"""CPU-only guardrails for the reused diagnostic metric, not model qualification."""
import sys
import unittest
import torch

sys.path.insert(0,'/workspace/r65-tools/migration_bridge/round6_5/tools')
from evaluate_static_pairwise import metrics


class FrozenMetricTest(unittest.TestCase):
    def test_identical_stable(self):
        n=torch.arange(64,dtype=torch.float32);n[63]=100
        r=metrics(n,n.clone(),63,63)
        self.assertTrue(r['stable'] and r['stable_exact'] and r['pass_gate'])

    def test_unchanged_top1_cannot_override_overflow(self):
        n=torch.arange(64,dtype=torch.float32);h=n.clone();h[61]-=3
        r=metrics(n,h,63,63)
        self.assertEqual(r['hybrid_top1'],r['baseline_top1'])
        self.assertGreater(r['overflow'],0)
        self.assertFalse(r['pass_gate'])

    def test_nonfinite_rejected(self):
        n=torch.arange(64,dtype=torch.float32);h=n.clone();h[0]=float('nan')
        with self.assertRaises(ValueError):metrics(n,h,63,63)

    def test_sampler_token_must_be_maximal(self):
        n=torch.arange(64,dtype=torch.float32)
        with self.assertRaises(ValueError):metrics(n,n.clone(),63,62)


if __name__=='__main__':unittest.main()
