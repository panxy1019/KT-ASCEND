import json
from pathlib import Path
import tempfile
import unittest

from summarize_final_screen import summarize, wilson


class FinalSummaryTests(unittest.TestCase):
    def test_finite_sample_uncertainty_is_nonzero_at_perfect_score(self):
        lo, hi = wilson(32, 32)
        self.assertLess(lo, .90)
        self.assertAlmostEqual(hi, 1.)

    def test_zero_score_does_not_imply_zero_population_rate(self):
        lo, hi = wilson(0, 32)
        self.assertAlmostEqual(lo, 0.)
        self.assertGreater(hi, .10)

    def test_incomplete_sequence_never_promotes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'progress.json').write_text(json.dumps({'status': 'RUNNING_QUALITY_NPU'}))
            result = summarize(root)
            self.assertFalse(result['final_qualification'])
            self.assertFalse(result['population_noninferiority_claim'])
            self.assertEqual(result['modes']['quality_hybrid']['status'], 'NOT_COMPLETE')


if __name__ == '__main__':
    unittest.main()
