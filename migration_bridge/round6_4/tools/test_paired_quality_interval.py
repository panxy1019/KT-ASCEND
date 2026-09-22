import unittest
import numpy as np
from scipy.stats import multinomial
from paired_quality_interval import bounds, evaluate


class PairedIntervalTests(unittest.TestCase):
    def test_zero_discordance_is_not_zero_uncertainty(self):
        result = evaluate([True]*1319, [True]*1319)
        self.assertEqual(result["bootstrap_percentile_95"], [0, 0])
        self.assertAlmostEqual(result["lower"], -(1 - .0125**(1/1319)))
        self.assertLess(result["lower"], 0)
        self.assertEqual(result["status"], "NONINFERIOR")

    def test_small_sample_does_not_get_free_pass(self):
        self.assertEqual(evaluate([True]*10, [True]*10)["status"], "INCONCLUSIVE")

    def test_clear_loss_and_gain(self):
        self.assertEqual(evaluate([True]*1000, [False]*1000)["status"], "REGRESSION_BEYOND_MARGIN")
        self.assertEqual(evaluate([False]*1000, [True]*1000)["status"], "NONINFERIOR")

    def test_symmetry_and_counts(self):
        a, b = bounds(200, 10, 20), bounds(200, 20, 10)
        self.assertAlmostEqual(a["lower"], -b["upper"])
        for args in ((0, 0, 0), (10, 8, 8), (10, -1, 0)):
            with self.assertRaises(ValueError):
                bounds(*args)

    def test_exact_finite_sample_coverage_grid(self):
        # Enumerate all trinomial outcomes, not a simulation of coverage.
        n = 20
        outcomes = np.array([(g, l, n-g-l) for g in range(n+1) for l in range(n-g+1)])
        lower = np.array([bounds(n, int(g), int(l))["lower"] for g, l, _ in outcomes])
        for pg in (0, .01, .1, .3, .5, .9):
            for pl in (0, .01, .1, .3, .5, .9):
                if pg+pl > 1:
                    continue
                pmf = multinomial.pmf(outcomes, n, [pg, pl, 1-pg-pl])
                self.assertLessEqual(float(pmf[lower > pg-pl+1e-12].sum()), .025+1e-12)


if __name__ == "__main__":
    unittest.main()
