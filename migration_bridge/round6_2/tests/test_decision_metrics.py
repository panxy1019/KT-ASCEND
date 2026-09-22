"""CPU-only tests of frozen decision rules; no model or held-out data."""
import sys
from pathlib import Path
import unittest
from unittest.mock import patch
import tempfile

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

    def test_exposure_gate_precedes_all_logit_reads(self):
        from evaluate_contract_v2 import main
        prompts = [{"id": f"p{i}"} for i in range(128)]
        rows = [dict(prompt_id=p["id"], token_index=t, deferred_event_count_current_row=0,
                     deferred_event_count_prefix=0) for p in prompts for t in range(64)]
        corpus = dict(corpus="QD2", sha256="frozen", prompts=prompts)
        manifests = [dict(mode=mode, corpus_sha256="frozen", row_count=8192,
                          complete_forward_count=16384,
                          baseline_history_sha256="same", rows=rows) for mode in ("defer0", "defer1")]
        with tempfile.TemporaryDirectory() as directory:
            argv = ["evaluate", "derive", "--baseline", directory, "--deferred", directory,
                    "--corpus", "unused", "--output", str(Path(directory) / "result")]
            with patch.object(sys, "argv", argv), patch("evaluate_contract_v2.read_frozen", side_effect=[corpus, *manifests]), patch("torch.load") as loader:
                with self.assertRaises(SystemExit) as caught:
                    main()
                self.assertEqual(caught.exception.code, 1)
                loader.assert_not_called()


if __name__ == "__main__":
    unittest.main()
