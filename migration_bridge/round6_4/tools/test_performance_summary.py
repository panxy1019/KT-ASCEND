import unittest
from summarize_performance import block_interval


class PerformanceTests(unittest.TestCase):
    def test_blocks_not_pooled_tokens(self):
        with self.assertRaises(ValueError):
            block_interval([1.1]*20)
        with self.assertRaises(ValueError):
            block_interval([1,1,1,1,0])

    def test_direction_and_uncertainty(self):
        x = block_interval([1.1,1.11,1.09,1.105,1.095])
        self.assertGreater(x["lower"],1)
        self.assertLess(x["lower"],x["geometric_mean"])
        self.assertGreater(x["upper"],x["geometric_mean"])
        self.assertFalse(block_interval([1.]*5)["variance_resolved"])

    def test_three_blocks_explicit(self):
        x=block_interval([1.10,1.11,1.09],3)
        self.assertGreater(x['lower'],1)
        self.assertEqual(x['method'],'student_t_log_ratio_df2_95')
        with self.assertRaises(ValueError):
            block_interval([1.1]*3)


if __name__ == "__main__":
    unittest.main()
