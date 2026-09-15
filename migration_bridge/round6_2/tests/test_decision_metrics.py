"""CPU-only tests of frozen decision rules; no model or held-out data."""
import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
try:
    import torch
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "torch unavailable locally; execute CPU-only in A3 container")
class DecisionTests(unittest.TestCase):
    def setUp(self):
        from evaluate_contract_v2 import metrics, classify
        self.metrics, self.classify = metrics, classify
        self.base = torch.full((102400,), -100.)
        self.base[0], self.base[1] = 10., 9.

    def row(self, deferred, chosen=0):
        row = self.metrics(self.base, deferred, 0, chosen)
        row["deferred_exposure"] = "CURRENT_EXPOSURE"
        return row

    def test_exact_stable(self):
        self.assertEqual(self.classify(self.row(self.base), .5)["failures"], [])

    def test_contraction_sign(self):
        deferred = self.base.clone()
        deferred[1] += .25
        self.assertEqual(self.row(deferred)["contraction"], .25)

    def test_beneficial_expansion_is_not_contraction(self):
        deferred = self.base.clone()
        deferred[1] -= .25
        self.assertEqual(self.row(deferred)["contraction"], 0)

    def test_stable_flip_and_allowed_ambiguity(self):
        deferred = self.base.clone()
        deferred[1] = 10.25
        row = self.row(deferred, 1)
        self.assertIn("STABLE_TOP1_MISMATCH", self.classify(row, .5)["failures"])
        self.assertEqual(self.classify(row, 1.25)["failures"], [])

    def test_no_exposure_is_byte_exact(self):
        deferred = self.base.clone()
        deferred[10] += .01
        row = self.row(deferred)
        row["deferred_exposure"] = "NO_EXPOSURE"
        self.assertIn("NO_EXPOSURE_LOGIT_MISMATCH", self.classify(row, .5)["failures"])

    def test_anchor_retained_and_universe_escape(self):
        self.base.fill_(0)
        deferred = self.base.clone()
        deferred[70] = 1.
        row = self.metrics(self.base, deferred, 64, 70)
        self.assertIn(64, row["candidate_ids"])
        self.assertEqual(len(row["candidate_ids"]), 33)
        row["deferred_exposure"] = "CURRENT_EXPOSURE"
        self.assertIn("OUTSIDE_CANDIDATE_UNIVERSE", self.classify(row, 5)["failures"])


if __name__ == "__main__":
    unittest.main()
