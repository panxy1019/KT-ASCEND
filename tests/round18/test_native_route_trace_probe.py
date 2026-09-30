from __future__ import annotations

import unittest

from round18_integration.native_route_trace_probe import build_route_trace_schedule


class NativeRouteTraceProbeTests(unittest.TestCase):
    def test_schedule_has_one_fresh_fixed_output_request_per_workload(self):
        fixture = {
            "cases": {
                "edge128": {"input_ids": list(range(128))},
                "edge64": {"input_ids": list(range(64))},
            }
        }
        schedule = build_route_trace_schedule(fixture)
        self.assertEqual([row["workload"] for row in schedule], ["W1", "W2", "W4"])
        self.assertEqual([row["input_tokens"] for row in schedule], [128, 128, 64])
        self.assertEqual([row["decode_tokens"] for row in schedule], [16, 128, 32])
        self.assertTrue(all(row["ignore_eos"] for row in schedule))

    def test_schedule_rejects_non_frozen_lengths(self):
        fixture = {
            "cases": {
                "edge128": {"input_ids": list(range(127))},
                "edge64": {"input_ids": list(range(64))},
            }
        }
        with self.assertRaisesRegex(RuntimeError, "length"):
            build_route_trace_schedule(fixture)


if __name__ == "__main__":
    unittest.main()
