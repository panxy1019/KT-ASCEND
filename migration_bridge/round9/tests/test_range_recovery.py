import hashlib
import sys
import tempfile
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))

class RecoveryTest(unittest.TestCase):
    def test_only_validated_range_is_appended(self):
        import range_recovery as recovery
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            target, part = root/'data', root/'part'
            target.write_bytes(b'ab')
            part.write_bytes(b'cd')
            with self.assertRaises(ValueError):
                recovery.append_range(target, part, 'HTTP/1.1 200 OK\r\n', 2, 3, 4)
            self.assertEqual(target.read_bytes(), b'ab')
            recovery.append_range(target, part, 'HTTP/1.1 206 OK\r\nContent-Range: bytes 2-3/4\r\n', 2, 3, 4)
            self.assertEqual(target.read_bytes(), b'abcd')

    def test_publication_requires_sha_and_never_overwrites(self):
        import range_recovery as recovery
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            stage, model = root/'stage', root/'model'
            stage.mkdir(); model.mkdir()
            (stage/'x').write_bytes(b'bad')
            row = dict(path='x', size=3, sha256=hashlib.sha256(b'yes').hexdigest())
            with self.assertRaises(ValueError):
                recovery.publish(stage, model, row)
            self.assertFalse((model/'x').exists())
            (stage/'x').write_bytes(b'yes')
            recovery.publish(stage, model, row)
            self.assertEqual((model/'x').read_bytes(), b'yes')
            with self.assertRaises(FileExistsError):
                recovery.publish(stage, model, row)
