from __future__ import annotations

import unittest

from round18_integration.native_hybrid_perf_probe import (
    build_schedule,
    summarize_workload,
)


class NativeHybridPerfProbeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.fixture = {
            "cases": {
                "edge64": {"input_ids": list(range(64))},
                "edge128": {"input_ids": list(range(128))},
                "canary_math391": {
                    "input_ids": [1, 2, 3],
                    "validator": {"kind": "exact_text", "expected": "391"},
                },
            }
        }

    def test_schedule_freezes_three_workloads_and_canary(self) -> None:
        schedule = build_schedule(self.fixture, "GRAPH-A", measurement_count=3)
        self.assertEqual(len(schedule), 13)
        for workload, input_tokens, decode_tokens in (
            ("W1", 128, 16),
            ("W2", 128, 128),
            ("W4", 64, 32),
        ):
            rows = [row for row in schedule if row.get("workload") == workload]
            self.assertEqual([row["phase"] for row in rows], ["warmup"] + ["measurement"] * 3)
            self.assertTrue(all(row["input_tokens"] == input_tokens for row in rows))
            self.assertTrue(all(row["decode_tokens"] == decode_tokens for row in rows))
        self.assertEqual(schedule[-1]["kind"], "canary")

    def test_summary_uses_continuous_measurement_window(self) -> None:
        rows = [
            {
                "phase": "warmup",
                "completion_tokens": 8,
                "started_monotonic_ns": 0,
                "ended_monotonic_ns": 1,
                "ttft_seconds": 9.0,
                "tpot_seconds": 9.0,
                "e2e_seconds": 9.0,
            },
            {
                "phase": "measurement",
                "completion_tokens": 16,
                "started_monotonic_ns": 1_000_000_000,
                "ended_monotonic_ns": 3_000_000_000,
                "ttft_seconds": 0.3,
                "tpot_seconds": 0.1,
                "e2e_seconds": 2.0,
            },
            {
                "phase": "measurement",
                "completion_tokens": 16,
                "started_monotonic_ns": 3_100_000_000,
                "ended_monotonic_ns": 5_000_000_000,
                "ttft_seconds": 0.2,
                "tpot_seconds": 0.09,
                "e2e_seconds": 1.9,
            },
        ]
        summary = summarize_workload(rows, expected_measurements=2)
        self.assertEqual(summary["measurement_requests"], 2)
        self.assertEqual(summary["output_tokens"], 32)
        self.assertAlmostEqual(summary["continuous_window_seconds"], 4.0)
        self.assertAlmostEqual(summary["serial_output_tokens_per_second"], 8.0)
        self.assertAlmostEqual(summary["requests_per_second"], 0.5)
        self.assertAlmostEqual(summary["median_tpot_seconds"], 0.095)

    def test_summary_rejects_missing_measurement(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "requires exactly 3"):
            summarize_workload([], expected_measurements=3)


if __name__ == "__main__":
    unittest.main()
