from pathlib import Path
import tempfile
import unittest

from tools.kt_perf_closure.analyze_profile import analyze, join_calls, load_jsonl


class AnalyzeProfileTests(unittest.TestCase):
    def test_native_reader_allows_only_one_terminal_partial_record(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trace.jsonl"
            path.write_text('{"schema":1,"event":"enqueue"}\n{"schema":1,"event":"task_complete"')
            diagnostics = {}
            rows = load_jsonl(
                path, allow_terminal_partial=True, diagnostics=diagnostics
            )
            self.assertEqual(len(rows), 1)
            self.assertEqual(diagnostics["terminal_partial_lines"], 1)

    def test_native_reader_rejects_partial_record_before_the_end(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trace.jsonl"
            path.write_text('{"schema":1\n{"schema":1,"event":"enqueue"}\n')
            with self.assertRaises(ValueError):
                load_jsonl(path, allow_terminal_partial=True)

    def test_native_cpu_and_host_boundaries_join(self):
        observer = [
            {"component": "HOST_DISPATCH", "parent_span_id": "p", "span_id": "d", "request_id": "x:W2_MEASUREMENT", "layer": 0, "start_ns": 110, "end_ns": 180, "duration_ns": 70},
            {"component": "HOST_SYNC", "parent_span_id": "p", "span_id": "s", "request_id": "x:W2_MEASUREMENT", "layer": 0, "start_ns": 250, "end_ns": 340, "duration_ns": 90},
            {"component": "HOST_MERGE", "parent_span_id": "p", "span_id": "m", "request_id": "x:W2_MEASUREMENT", "layer": 0, "start_ns": 350, "end_ns": 360, "duration_ns": 10},
            {"component": "PARENT_ROUTED_EXPERT", "parent_span_id": "batch", "span_id": "p", "request_id": "x:W2_MEASUREMENT", "layer": 0, "start_ns": 100, "end_ns": 380, "duration_ns": 280},
        ]
        native = [
            {"event": "enqueue", "sequence": 7, "now_ns": 160, "admitted_ns": 160, "start_ns": 0, "pending": 1, "allow_pending": 0},
            {"event": "task_complete", "sequence": 7, "now_ns": 315, "admitted_ns": 160, "start_ns": 170, "pending": 1, "allow_pending": 0},
            {"event": "sync_begin", "sequence": 3, "now_ns": 300, "admitted_ns": 0, "start_ns": 0, "pending": 1, "allow_pending": 0},
            {"event": "sync_end", "sequence": 3, "now_ns": 320, "admitted_ns": 0, "start_ns": 300, "pending": 0, "allow_pending": 0},
        ]
        round13 = [{"status": "ROUND13_P0_BATCH", "request_id": "x:W2_MEASUREMENT", "mode": "decode", "batch_sequence": 2, "started_ns": 50, "ended_ns": 500, "duration_ns": 450, "spans": []}]
        rows = join_calls(observer, native, round13)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["queue_wait_ns"], 10)
        self.assertEqual(rows[0]["cpu_task_ns"], 145)
        self.assertEqual(rows[0]["host_sync_pre_native_ns"], 50)
        self.assertEqual(rows[0]["native_sync_wait_ns"], 20)
        self.assertEqual(rows[0]["cpu_tail_after_host_sync_start_ns"], 65)
        self.assertEqual(rows[0]["host_sync_start_ns"], 250)
        self.assertEqual(rows[0]["native_sync_begin_ns"], 300)
        self.assertEqual(rows[0]["cpu_hidden_before_host_sync_ns"], 0)


if __name__ == "__main__":
    unittest.main()
