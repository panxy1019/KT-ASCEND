from __future__ import annotations

import unittest

from round16_integration.b_v5_perf_probe import build_schedule, summarize_workload


class V5PerformanceProbeTests(unittest.TestCase):
    def test_schedule_has_three_fixed_workloads_and_one_canary(self):
        fixture = {
            "cases": {
                "edge128": {"input_ids": list(range(128))},
                "edge64": {"input_ids": list(range(64))},
                "canary_math391": {
                    "input_ids": list(range(27)),
                    "validator": {"kind": "exact_text", "expected": "391"},
                },
            }
        }
        schedule = build_schedule(fixture, "B-A")
        self.assertEqual(len(schedule), 10)
        self.assertEqual(
            [(row["workload"], row["phase"]) for row in schedule[:9]],
            [
                ("W1", "warmup"),
                ("W1", "measurement"),
                ("W1", "measurement"),
                ("W2", "warmup"),
                ("W2", "measurement"),
                ("W2", "measurement"),
                ("W4", "warmup"),
                ("W4", "measurement"),
                ("W4", "measurement"),
            ],
        )
        self.assertEqual(schedule[-1]["kind"], "canary")
        self.assertEqual(schedule[0]["input_tokens"], 128)
        self.assertEqual(schedule[0]["decode_tokens"], 16)
        self.assertEqual(schedule[3]["decode_tokens"], 128)
        self.assertEqual(schedule[6]["input_tokens"], 64)
        self.assertEqual(schedule[6]["decode_tokens"], 32)

    def test_summary_uses_complete_serial_request_window(self):
        rows = [
            {"phase": "warmup", "e2e_seconds": 100.0, "completion_tokens": 16},
            {
                "phase": "measurement",
                "e2e_seconds": 4.0,
                "completion_tokens": 16,
                "ttft_seconds": 1.0,
                "tpot_seconds": 0.2,
            },
            {
                "phase": "measurement",
                "e2e_seconds": 6.0,
                "completion_tokens": 16,
                "ttft_seconds": 2.0,
                "tpot_seconds": 0.3,
            },
        ]
        summary = summarize_workload(rows)
        self.assertEqual(summary["measurement_requests"], 2)
        self.assertEqual(summary["window_seconds"], 10.0)
        self.assertEqual(summary["output_tokens"], 32)
        self.assertAlmostEqual(summary["serial_output_tokens_per_second"], 3.2)
        self.assertAlmostEqual(summary["requests_per_second"], 0.2)
        self.assertEqual(summary["median_ttft_seconds"], 1.5)
        self.assertEqual(summary["median_tpot_seconds"], 0.25)


if __name__ == "__main__":
    unittest.main()
