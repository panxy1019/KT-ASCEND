import importlib.util
import unittest
from evaluate_static_pairwise import metrics


@unittest.skipUnless(importlib.util.find_spec('torch'),'torch required; run in target container')
class Pairwise(unittest.TestCase):
    def vector(self):
        import torch
        n=torch.arange(128,dtype=torch.float32)*-1
        n[0]=10
        return n

    def test_stable_exact(self):
        n=self.vector()
        self.assertTrue(metrics(n,n.clone(),0)['pass_gate'])
        self.assertTrue(metrics(n,n.clone(),0)['stable'])

    def test_overflow_rejected_without_token_flip(self):
        n=self.vector();h=n.clone();h[1]-=3
        self.assertFalse(metrics(n,h,0)['pass_gate'])

    def test_ambiguity_allowed(self):
        n=self.vector();n[1]=9.5;h=n.clone();h[1]=10.25
        r=metrics(n,h,0)
        self.assertFalse(r['stable']);self.assertTrue(r['pass_gate'])

    def test_stable_flip(self):
        n=self.vector();h=n.clone();h[1]=11
        self.assertFalse(metrics(n,h,0)['pass_gate'])

    def test_nonfinite(self):
        n=self.vector();h=n.clone();h[2]=float('nan')
        with self.assertRaises(ValueError):metrics(n,h,0)

    def test_new_candidate_in_union(self):
        n=self.vector();h=n.clone();h[80]=12
        self.assertGreater(metrics(n,h,0)['overflow'],0)

    def test_actual_sampler_tie_choice(self):
        n=self.vector();n[1]=9.5;h=n.clone();h[1]=h[0]
        self.assertEqual(metrics(n,h,0,1)['hybrid_top1'],1)
        self.assertTrue(metrics(n,h,0,1)['pass_gate'])

    def test_sampler_must_match_exported_logits(self):
        n=self.vector()
        with self.assertRaisesRegex(ValueError,'SERVING_TOKEN_NOT_MAX'):metrics(n,n,0,2)


if __name__=='__main__':unittest.main()
