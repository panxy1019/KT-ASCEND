import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch


class WorkingSetTimelineTests(unittest.TestCase):
    def tearDown(self):
        from migration_bridge.round9.runtime.working_set_timeline import reset_for_tests
        reset_for_tests()

    def test_default_off_has_no_state_or_file(self):
        from migration_bridge.round9.runtime.working_set_timeline import (
            begin_request, event, flush_request, snapshot,
        )
        with tempfile.TemporaryDirectory() as directory, patch.dict(
                os.environ, {}, clear=True):
            path = Path(directory) / "timeline.jsonl"
            self.assertFalse(begin_request("r", 0, 0))
            self.assertFalse(event("scheduler", "chunk_begin"))
            self.assertFalse(flush_request("complete"))
            self.assertIsNone(snapshot())
            self.assertFalse(path.exists())

    def test_bounded_batch_flush_reports_drops_and_clock_domains(self):
        from migration_bridge.round9.runtime.working_set_timeline import (
            begin_request, event, flush_request,
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "timeline.jsonl"
            with patch.dict(os.environ, {
                    "ROUND9_TIMELINE_PATH": str(path),
                    "ROUND9_TIMELINE_MAX_EVENTS": "3"}, clear=True):
                self.assertTrue(begin_request("r", 2, 7))
                self.assertTrue(event("scheduler", "chunk_begin", start=0, count=32))
                self.assertTrue(event(
                    "ple", "h2d_enqueue",
                    clock_domain="HOST_OBSERVED_DEVICE_OPERATION", bytes=128))
                self.assertTrue(event("expert", "shadow_prepare", layer=0))
                self.assertFalse(event("expert", "compute", layer=0))
                self.assertTrue(flush_request("complete"))
            rows = [json.loads(line) for line in path.read_text().splitlines()]
            self.assertEqual(len(rows), 1)
            row = rows[0]
            self.assertEqual(row["identity"], {
                "request_id": "r", "placement_epoch": 2, "allocation_epoch": 7})
            self.assertEqual(row["event_count"], 3)
            self.assertEqual(row["dropped_events"], 1)
            self.assertEqual([item["sequence"] for item in row["events"]], [0, 1, 2])
            self.assertEqual(row["events"][1]["clock_domain"],
                             "HOST_OBSERVED_DEVICE_OPERATION")

    def test_identity_switch_and_invalid_clock_fail_closed(self):
        from migration_bridge.round9.runtime.working_set_timeline import begin_request, event
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {
                "ROUND9_TIMELINE_PATH": str(Path(directory) / "timeline.jsonl")}, clear=True):
            begin_request("r", 0, 1)
            with self.assertRaises(RuntimeError):
                begin_request("other", 0, 2)
            with self.assertRaises(ValueError):
                event("ple", "wait", clock_domain="NPU_MADE_UP")

    def test_enabled_timeline_ignores_unprofiled_request_flush(self):
        from migration_bridge.round9.runtime.working_set_timeline import flush_request
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "timeline.jsonl"
            with patch.dict(os.environ, {
                    "ROUND9_TIMELINE_PATH": str(path)}, clear=True):
                self.assertFalse(flush_request("abort"))
            self.assertFalse(path.exists())


if __name__ == "__main__":
    unittest.main()
