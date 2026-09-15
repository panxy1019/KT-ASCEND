import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from capture_session_boundary import require_empty_capture_session


class CaptureSessionBoundaryTest(unittest.TestCase):
    def test_empty(self):
        with tempfile.TemporaryDirectory() as tmp:
            require_empty_capture_session(Path(tmp))

    def test_existing_forward_rejected_not_deleted(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            nested = root / 'rank'
            nested.mkdir()
            prior = nested / 'Pass00000.pt'
            prior.touch()
            with self.assertRaisesRegex(ValueError, 'CAPTURE_SESSION_NOT_EMPTY'):
                require_empty_capture_session(root)
            self.assertTrue(prior.exists())

    def test_incomplete_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'Pass00000.pt.incomplete').touch()
            with self.assertRaisesRegex(ValueError, 'CAPTURE_SESSION_NOT_EMPTY'):
                require_empty_capture_session(root)

    def test_missing_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError, 'CAPTURE_DIRECTORY_NOT_READY'):
                require_empty_capture_session(Path(tmp) / 'missing')


if __name__ == '__main__':
    unittest.main()
