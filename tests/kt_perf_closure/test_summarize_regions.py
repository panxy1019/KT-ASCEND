import unittest

from tools.kt_perf_closure.summarize_regions import analyze_sampled_spans


class SummarizeRegionsTests(unittest.TestCase):
    def test_qsa_parent_excludes_nested_indexer_from_scaled_sum(self):
        rows = [
            {
                "status": "ROUND13_P0_BATCH",
                "request_id": "x:W2_MEASUREMENT",
                "mode": "decode",
                "batch_sequence": 9,
                "dropped_spans": 0,
                "observer_errors": [],
                "spans": [
                    {
                        "component": "QSA_ATTENTION",
                        "layer": 3,
                        "duration_ns": 10_000_000,
                        "device_ms": 10.5,
                        "device_status": "COMPLETE",
                    },
                    {
                        "component": "QSA_INDEXER",
                        "layer": 3,
                        "duration_ns": 4_000_000,
                        "device_ms": 4.5,
                        "device_status": "COMPLETE",
                    },
                    {
                        "component": "GDN",
                        "layer": 0,
                        "duration_ns": 2_000_000,
                        "device_ms": 2.5,
                        "device_status": "COMPLETE",
                    },
                ],
            }
        ]
        result = analyze_sampled_spans(rows, warm_decode_batches=0)
        self.assertEqual(result["components"]["QSA_INDEXER"]["nested_in"], "QSA_ATTENTION")
        self.assertEqual(result["scaled_nonadditive_estimates_ms_per_token"]["QSA_ATTENTION"]["median_ms"], 120.0)
        self.assertEqual(result["scaled_nonadditive_estimates_ms_per_token"]["GDN"]["median_ms"], 72.0)
        self.assertNotIn("QSA_INDEXER", result["scaled_nonadditive_estimates_ms_per_token"])
        self.assertTrue(result["scaled_estimates_are_nonadditive"])


if __name__ == "__main__":
    unittest.main()
