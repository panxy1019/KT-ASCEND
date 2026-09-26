import io
import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import round10_t1_driver as driver


class DriverHelpersTests(unittest.TestCase):
    def test_jsonl_reader_rejects_unterminated_or_malformed_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "events.jsonl"
            path.write_text('{"ok":true}')
            with self.assertRaisesRegex(RuntimeError, "unterminated"):
                driver.read_jsonl(path)
            path.write_text("not-json\n")
            with self.assertRaisesRegex(RuntimeError, "malformed"):
                driver.read_jsonl(path)

    def test_cancel_summary_detects_late_normal_submit(self):
        row = {"request": {"rid": "cancel"}}
        response = [{"meta_info": {"finish_reason": {"type": "abort"}}}]
        hybrid = [
            {"request_id": "cancel", "stage": "SELECTION", "start": 0},
            {"request_id": "cancel", "stage": "SELECTION", "start": 32},
        ]
        events = [
            {"request_id": "cancel", "status": "CANCEL_SEEN", "event_seq": 5},
            {"request_id": "cancel", "status": "NORMAL_SUBMIT_ADMITTED", "event_seq": 6},
            {"request_id": "cancel", "status": "REQUEST_RETIRED", "event_seq": 7},
            {"request_id": "cancel", "status": "CANCEL_CLEANUP_COMMITTED", "event_seq": 8},
        ]
        got = driver.cancel_summary(
            row,
            response,
            hybrid,
            events,
            abort_http_status=200,
            thread_alive=False,
        )
        self.assertTrue(got["late_commit"])
        events.pop(1)
        got = driver.cancel_summary(
            row,
            response,
            hybrid,
            events,
            abort_http_status=200,
            thread_alive=False,
        )
        self.assertFalse(got["late_commit"])
        self.assertTrue(got["lease_returned"])


if __name__ == "__main__":
    unittest.main()
