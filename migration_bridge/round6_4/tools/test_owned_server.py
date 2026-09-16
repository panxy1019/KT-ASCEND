import json
from pathlib import Path
import signal
import tempfile
import unittest
from unittest.mock import Mock
from owned_server import OwnedServer


class OwnedServerTests(unittest.TestCase):
    def test_scope_rejection(self):
        for profile, batch in (("D3", 1), ("D1", 8)):
            with self.assertRaises(ValueError):
                OwnedServer("/workspace/results/r64-test", profile, batch)

    def test_only_owned_guard_terminated(self):
        with tempfile.TemporaryDirectory() as d:
            server = OwnedServer("/workspace/results/r64-test", "D1")
            server.root = Path(d)
            (server.root/"guard").mkdir()
            (server.root/"guard/result.json").write_text(json.dumps(dict(reason="CANCELLED")))
            process = Mock()
            process.poll.return_value = None
            server.server = process
            server.close()
            process.send_signal.assert_called_once_with(signal.SIGTERM)
            process.wait.assert_called_once_with(timeout=60)
            self.assertEqual(json.loads((server.root/"lifecycle.json").read_text())["status"], "PLANNED_STOP")

    def test_unexpected_exit_not_pass(self):
        with tempfile.TemporaryDirectory() as d:
            server = OwnedServer("/workspace/results/r64-test", "D1")
            server.root = Path(d)
            (server.root/"guard").mkdir()
            (server.root/"guard/result.json").write_text(json.dumps(dict(reason="ERROR_PATTERN")))
            process = Mock()
            process.poll.return_value = 1
            server.server = process
            with self.assertRaisesRegex(RuntimeError, "UNEXPECTED"):
                server.close()
            process.send_signal.assert_not_called()
            self.assertEqual(json.loads((server.root/"lifecycle.json").read_text())["status"], "FAILED")


if __name__ == "__main__":
    unittest.main()
