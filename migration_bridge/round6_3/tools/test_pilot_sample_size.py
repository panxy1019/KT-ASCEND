import unittest
from pilot_sample_size import choose


class SampleSizeTests(unittest.TestCase):
    def test_no_exposure_does_not_expand_until_pass(self):
        self.assertIsNone(choose([0] * 32)["selected_prompts"])

    def test_high_exposure_can_support_smallest_frozen_size(self):
        self.assertEqual(choose([64] * 32)["selected_prompts"], 128)

    def test_invalid_blocks_rejected(self):
        for data in ([64] * 31, [65] * 32, [-1] * 32, [True] * 32):
            with self.assertRaises(ValueError):
                choose(data)

    def test_rule_is_order_invariant(self):
        data = [i * 2 for i in range(32)]
        self.assertEqual(choose(data), choose(list(reversed(data))))


if __name__ == "__main__":
    unittest.main()
