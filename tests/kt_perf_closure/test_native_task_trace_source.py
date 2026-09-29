from pathlib import Path
import tempfile
import unittest

from tools.kt_perf_closure.build_native_task_trace_source import (
    patch_header,
    patch_source,
)


ROOT = Path(__file__).resolve().parents[2]


class NativeTaskTraceSourceTests(unittest.TestCase):
    def test_frozen_sources_patch_once(self):
        header = (ROOT / "kt-kernel/cpu_backend/task_queue.h").read_text()
        source = (ROOT / "kt-kernel/cpu_backend/task_queue.cpp").read_text()
        patched_header = patch_header(header)
        patched_source = patch_source(source)
        self.assertIn("uint64_t trace_sequence", patched_header)
        self.assertIn("KT_ROOT_TASK_TRACE_PATH", patched_source)
        self.assertIn('"task_complete"', patched_source)
        self.assertIn('"sync_begin"', patched_source)
        self.assertIn('"sync_end"', patched_source)
        self.assertEqual(patched_source.count("TaskQueue::sync("), 1)

    def test_patch_refuses_already_patched_source(self):
        source = (ROOT / "kt-kernel/cpu_backend/task_queue.cpp").read_text()
        with self.assertRaises(ValueError):
            patch_source(patch_source(source))


if __name__ == "__main__":
    unittest.main()
