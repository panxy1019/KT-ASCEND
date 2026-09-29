import unittest

from round16_integration.compare_request_blocks import compare_blocks


def block(name, tpot, throughput, *, e2e=10.0, ttft=2.0):
    status = (
        "PASS_ROUND16_S0_REQUEST_PERFORMANCE_BLOCK"
        if name.startswith("S0")
        else "PASS_ROUND16_B_REQUEST_PERFORMANCE_BLOCK"
    )
    results = []
    for workload, decode in (("W1", 16), ("W2", 128), ("W4", 32)):
        for repeat in (1, 2):
            results.append(
                {
                    "kind": "performance",
                    "workload": workload,
                    "phase": "measurement",
                    "repeat": repeat,
                    "decode_tokens": decode,
                    "completion_tokens": decode,
                    "token_ids": [3897, 248046] + [1] * (decode - 2),
                }
            )
    return {
        "status": status,
        "block": name,
        "profiler": "OFF",
        "generation_attempts": 10,
        "automatic_retries": 0,
        "results": results,
        "summaries": {
            workload: {
                "measurement_requests": 2,
                "output_tokens": decode * 2,
                "median_tpot_seconds": tpot,
                "serial_output_tokens_per_second": throughput,
                "median_e2e_seconds": e2e,
                "median_ttft_seconds": ttft,
            }
            for workload, decode in (("W1", 16), ("W2", 128), ("W4", 32))
        },
    }


class CompareRequestBlocksTests(unittest.TestCase):
    def test_two_slower_candidate_blocks_close_without_promotion(self):
        result = compare_blocks(
            block("S0-A", 0.25, 4.0),
            block("B-A", 0.31, 3.1, e2e=11.0),
            block("B-B", 0.30, 3.2, e2e=10.8),
            block("S0-B", 0.26, 3.8),
        )
        self.assertEqual(result["disposition"], "B16_MEASURED_NO_PROMOTION")
        self.assertFalse(result["decode_gate_pass"])
        self.assertAlmostEqual(result["pairs"][0]["W2"]["tpot_ratio"], 1.24)

    def test_rejects_shortened_performance_output(self):
        candidate = block("B-A", 0.2, 5.0)
        candidate["results"][0]["completion_tokens"] = 15
        with self.assertRaisesRegex(ValueError, "output budget"):
            compare_blocks(
                block("S0-A", 0.25, 4.0),
                candidate,
                block("B-B", 0.2, 5.0),
                block("S0-B", 0.25, 4.0),
            )


if __name__ == "__main__":
    unittest.main()
