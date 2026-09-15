import importlib.util
import math
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location(
    "tail_statistics", Path(__file__).resolve().parents[1] / "tools/tail_statistics.py")
stats = importlib.util.module_from_spec(spec)
spec.loader.exec_module(stats)


class TailStatisticsTests(unittest.TestCase):
    def test_small_binomial_exact(self):
        self.assertAlmostEqual(stats.binomial_cdf(2, 4, .5), 11 / 16)

    def test_zero_failure_closed_form(self):
        self.assertAlmostEqual(stats.upper_confidence_bound(0, 128, .05), 1 - .05 ** (1 / 128))

    def test_nist_reference(self):
        # NIST two-sided 90% upper endpoint = one-sided 95% endpoint.
        upper = stats.upper_confidence_bound(4, 20, .05)
        self.assertAlmostEqual(upper, .401029, delta=1e-6)
        self.assertAlmostEqual(stats.binomial_cdf(4, 20, upper), .05, places=12)

    def test_rank_is_minimal(self):
        rank = stats.tolerance_rank(128, .95, .05 / 3)
        self.assertEqual(rank, 127)
        self.assertGreaterEqual(stats.binomial_cdf(rank - 1, 128, .95), 1 - .05 / 3)
        self.assertLess(stats.binomial_cdf(rank - 2, 128, .95), 1 - .05 / 3)

    def test_insufficient_units(self):
        with self.assertRaisesRegex(ValueError, "INSUFFICIENT"):
            stats.tolerance_rank(16, .995, .05)

    def test_nonzero_tail_can_pass(self):
        result = stats.summarize_tail([2] + [0] * 199, 1, .05, .05)
        self.assertTrue(result["pass"])
        self.assertEqual(result["exceedances"], 1)

    def test_zero_tail_not_enough_when_small_n(self):
        self.assertFalse(stats.summarize_tail([0] * 10, 1, .05, .05)["pass"])

    def test_invalid_values_rejected(self):
        for value in (math.nan, math.inf, -1):
            with self.assertRaises(ValueError):
                stats.derive_threshold([value], .95, .05)

    def test_prompt_not_pair_unit(self):
        rows = [{"prompt_id": "a", "token_index": i, "risk": i} for i in range(4)]
        self.assertEqual(stats.prompt_block_maxima(rows, "risk", {"a": 4}), [3])
        with self.assertRaisesRegex(ValueError, "duplicate"):
            stats.prompt_block_maxima(rows + rows[:1], "risk", {"a": 4})
        with self.assertRaisesRegex(ValueError, "missing"):
            stats.prompt_block_maxima(rows[:-1], "risk", {"a": 4})


if __name__ == "__main__":
    unittest.main()
