import unittest
from summarize_primary_quality import stratified_bootstrap


class StratifiedDiagnosticTests(unittest.TestCase):
    def test_zero_discordance_supplement_not_inference(self):
        self.assertEqual(stratified_bootstrap([0]*8, ["a"]*4+["b"]*4), [0.,0.])

    def test_fixed_stratum_counts_and_seed(self):
        self.assertEqual(stratified_bootstrap([1,1,-1,-1], ["a","a","b","b"]), [0.,0.])
        args = ([0,1,0,-1,0,0], ["a"]*3+["b"]*3)
        self.assertEqual(stratified_bootstrap(*args), stratified_bootstrap(*args))


if __name__ == "__main__":
    unittest.main()
