import importlib.util
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

spec = importlib.util.spec_from_file_location("guard", Path(__file__).resolve().parents[1] / "tools/run_owned_process.py")
guard = importlib.util.module_from_spec(spec)
spec.loader.exec_module(guard)


class OwnedProcessTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.output = Path(self.tmp.name) / "run"

    def test_clean_exit(self):
        result = guard.run([sys.executable, "-c", "print('done')"], self.output, 5, grace=.01, interval=.01)
        self.assertTrue(result["success"])
        self.assertEqual(result["signals_sent"], ["SIGTERM", "SIGKILL"])

    def test_error_only_kills_owned_process(self):
        sentinel = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"], start_new_session=True)
        try:
            result = guard.run([sys.executable, "-u", "-c", "import time; print('Out of memory'); time.sleep(30)"], self.output, 5, grace=.2, interval=.01)
            self.assertEqual(result["reason"], "ERROR_LOG")
            self.assertFalse(result["success"])
            self.assertIn("SIGTERM", result["signals_sent"])
            self.assertIsNone(sentinel.poll())
        finally:
            sentinel.terminate()
            sentinel.wait()

    def test_timeout(self):
        result = guard.run([sys.executable, "-c", "import time; time.sleep(30)"], self.output, .1, grace=.2, interval=.01)
        self.assertEqual(result["reason"], "TIMEOUT")
        self.assertFalse(result["success"])

    def test_refuse_evidence_overwrite(self):
        self.output.mkdir()
        with self.assertRaises(FileExistsError):
            guard.run([sys.executable, "-c", "pass"], self.output, 1)

    def test_nonzero_exit(self):
        result = guard.run([sys.executable, "-c", "raise SystemExit(7)"], self.output, 5, grace=.01, interval=.01)
        self.assertEqual(result["reason"], "EXIT_NONZERO")


if __name__ == "__main__":
    unittest.main()
