import os
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import targeted_cache_retirement as retirement


class TargetedRetirementTests(unittest.TestCase):
    def test_open_target_file_is_detected_and_stat_identity_is_stable(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "weight.safetensors"
            path.write_bytes(b"weights")
            before = retirement.stat_identity([path])
            with path.open("rb"):
                references = retirement.scan_process_references([root], pids=[os.getpid()])
                self.assertTrue(
                    any(item["pid"] == os.getpid() and item["kind"] == "fd" for item in references)
                )
            self.assertEqual(before, retirement.stat_identity([path]))

    def test_open_symlink_target_is_detected_from_exact_candidate_files(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            candidate = base / "candidate"
            backing = base / "backing"
            candidate.mkdir()
            backing.mkdir()
            target = backing / "layer.gguf"
            target.write_bytes(b"weights")
            link = candidate / "layer.gguf"
            link.symlink_to(target)
            with target.open("rb"):
                references = retirement.scan_process_references(
                    [candidate],
                    exact_files=[link],
                    pids=[os.getpid()],
                )
                self.assertTrue(
                    any(item["pid"] == os.getpid() and item["kind"] == "fd" for item in references)
                )


if __name__ == "__main__":
    unittest.main()
