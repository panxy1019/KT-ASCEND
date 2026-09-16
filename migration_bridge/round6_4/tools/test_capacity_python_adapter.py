import unittest
from capacity_python_adapter import adapt


class CapacityTests(unittest.TestCase):
    def setUp(self):
        self.argv = ["-m", "sglang.launch_server", "--context-length", "512",
                     "--max-total-tokens", "512", "--chunked-prefill-size", "512",
                     "--max-prefill-tokens", "512", "--max-running-requests", "1",
                     "--tp-size", "1", "--disable-cuda-graph", "--kt-method", "LLAMAFILE",
                     "--kt-max-deferred-experts-per-token", "1"]

    def test_only_capacity_changes(self):
        for batch in (1, 2, 4):
            got = adapt(self.argv, batch)
            self.assertEqual(got[-7:], self.argv[-7:])
            self.assertEqual(got[got.index("--max-running-requests")+1], str(batch))
            self.assertEqual(got[got.index("--context-length")+1], "1024")
            self.assertEqual(self.argv[3], "512")

    def test_no_duplicate_or_unexpected_options(self):
        with self.assertRaises(ValueError):
            adapt(self.argv + ["--context-length", "512"], 1)
        with self.assertRaises(ValueError):
            adapt(self.argv, 8)
        self.argv[3] = "1024"
        with self.assertRaises(ValueError):
            adapt(self.argv, 1)


if __name__ == "__main__":
    unittest.main()
