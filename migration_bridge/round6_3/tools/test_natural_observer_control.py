import unittest
from compare_natural_observers import compare_rows


class NaturalParity(unittest.TestCase):
    def test_exact(self):
        a = [dict(rid='r', input_ids=[1], baseline_output_ids=[2, 3])]
        compare_rows(a, a)

    def test_flip_rejected(self):
        a = [dict(rid='r', input_ids=[1], baseline_output_ids=[2, 3])]
        b = [dict(rid='r', input_ids=[1], baseline_output_ids=[2, 4])]
        with self.assertRaisesRegex(AssertionError, 'TOKEN_PARITY'):
            compare_rows(a, b)

    def test_history_rejected(self):
        a = [dict(rid='r', input_ids=[1], baseline_output_ids=[2])]
        b = [dict(rid='r', input_ids=[3], baseline_output_ids=[2])]
        with self.assertRaisesRegex(AssertionError, 'INPUT_IDENTITY'):
            compare_rows(a, b)


if __name__ == '__main__':
    unittest.main()
