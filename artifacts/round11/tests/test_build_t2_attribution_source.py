import unittest
from pathlib import Path

from artifacts.round11.build_t2_attribution_source import patch_header, patch_source


ROOT = Path(__file__).resolve().parents[3]


class BuildT2AttributionSourceTests(unittest.TestCase):
    def test_task_queue_patch_is_strict_and_env_gated(self):
        header = (ROOT / "kt-kernel/cpu_backend/task_queue.h").read_text()
        source = (ROOT / "kt-kernel/cpu_backend/task_queue.cpp").read_text()
        changed_header = patch_header(header)
        changed_source = patch_source(source)
        self.assertIn("uint64_t admitted_ns", changed_header)
        self.assertIn("KT_TASK_QUEUE_TIMING", changed_source)
        self.assertIn("queue_admission_wait", changed_source)
        self.assertIn("task_execute", changed_source)
        self.assertNotEqual(header, changed_header)
        self.assertNotEqual(source, changed_source)

    def test_patch_refuses_second_application(self):
        header = (ROOT / "kt-kernel/cpu_backend/task_queue.h").read_text()
        source = (ROOT / "kt-kernel/cpu_backend/task_queue.cpp").read_text()
        with self.assertRaisesRegex(ValueError, "precondition"):
            patch_header(patch_header(header))
        with self.assertRaisesRegex(ValueError, "precondition"):
            patch_source(patch_source(source))


if __name__ == "__main__":
    unittest.main()
