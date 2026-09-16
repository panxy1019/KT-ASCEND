import unittest
from audit_batch_capture import expected_row


class BatchHistoryTests(unittest.TestCase):
    def test_prefill_decode_and_lookahead(self):
        request = dict(input_ids=[10,11,12], output_ids=[20,21])
        self.assertEqual(expected_row(request,0), ([10,11,12],[0,1,2],3))
        self.assertEqual(expected_row(request,1), ([20],[3],4))
        self.assertEqual(expected_row(request,2), ([21],[4],5))
        with self.assertRaises(ValueError):
            expected_row(request,3)


if __name__ == "__main__":
    unittest.main()
