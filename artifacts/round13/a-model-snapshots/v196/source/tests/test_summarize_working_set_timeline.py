import json
from pathlib import Path
import tempfile
import unittest


class TimelineSummaryTests(unittest.TestCase):
    def test_reports_waits_overlap_and_repeated_shadow_candidates(self):
        from migration_bridge.round9.tools.summarize_working_set_timeline import summarize
        events = [
            {"sequence": 0, "host_monotonic_ns": 10, "clock_domain": "HOST_MONOTONIC",
             "category": "scheduler", "phase": "batch_begin", "mode": "extend"},
            {"sequence": 1, "host_monotonic_ns": 20, "clock_domain": "HOST_MONOTONIC",
             "category": "ple", "phase": "gather_submit", "duration_ns": 3, "bytes": 64},
            {"sequence": 2, "host_monotonic_ns": 30,
             "clock_domain": "HOST_OBSERVED_DEVICE_OPERATION", "category": "expert",
             "phase": "shadow_prepare", "duration_ns": 11, "layer": 0,
             "logical_experts": [1, 2], "bytes": 128},
            {"sequence": 3, "host_monotonic_ns": 40,
             "clock_domain": "HOST_MONOTONIC", "category": "ple",
             "phase": "gather_wait_and_h2d_dependency", "duration_ns": 7},
            {"sequence": 4, "host_monotonic_ns": 50,
             "clock_domain": "HOST_OBSERVED_DEVICE_OPERATION", "category": "ple",
             "phase": "h2d_enqueue", "duration_ns": 2, "bytes": 64},
            {"sequence": 5, "host_monotonic_ns": 55,
             "clock_domain": "HOST_OBSERVED_DEVICE_OPERATION", "category": "expert",
             "phase": "shadow_prepare", "duration_ns": 13, "layer": 0,
             "logical_experts": [1, 2], "bytes": 128},
            {"sequence": 6, "host_monotonic_ns": 58,
             "clock_domain": "HOST_OBSERVED_DEVICE_OPERATION", "category": "ple",
             "phase": "transfer_consumer_retire_wait", "duration_ns": 3, "bytes": 64},
            {"sequence": 7, "host_monotonic_ns": 60, "clock_domain": "HOST_MONOTONIC",
             "category": "scheduler", "phase": "batch_end", "mode": "extend"},
        ]
        document = {
            "schema": 1, "identity": {"request_id": "r"}, "event_count": len(events),
            "dropped_events": 0, "events": events,
        }
        summary = summarize([document])
        self.assertEqual(summary["status"], "PASS_BOUNDED_TIMELINE_PROFILE")
        self.assertEqual(summary["phase_metrics"]["ple:gather_wait_and_h2d_dependency"]["total_ns"], 7)
        self.assertTrue(summary["independent_work_before_ple_wait_observed"])
        self.assertTrue(summary["h2d_expert_compute_overlap_observed"])
        self.assertEqual(summary["repeated_shadow_identity_candidates"], 1)
        self.assertEqual(summary["scheduler_batches"][0]["duration_ns"], 50)

    def test_dropped_or_missing_plane_is_incomplete(self):
        from migration_bridge.round9.tools.summarize_working_set_timeline import summarize
        summary = summarize([{
            "schema": 1, "identity": {"request_id": "r"}, "event_count": 1,
            "dropped_events": 2,
            "events": [{"sequence": 0, "host_monotonic_ns": 1,
                        "clock_domain": "HOST_MONOTONIC", "category": "scheduler",
                        "phase": "batch_begin"}],
        }])
        self.assertEqual(summary["status"], "INCOMPLETE_BOUNDED_TIMELINE_PROFILE")
        self.assertEqual(summary["dropped_events"], 2)


if __name__ == "__main__":
    unittest.main()
