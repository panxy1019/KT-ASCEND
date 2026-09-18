import hashlib
import importlib.util
from pathlib import Path
import tempfile
import unittest

S=importlib.util.spec_from_file_location('verify_checkpoint',Path(__file__).resolve().parents[1]/'tools/verify_checkpoint.py')
M=importlib.util.module_from_spec(S);S.loader.exec_module(M)


class CheckpointIdentityTests(unittest.TestCase):
    def test_identity_and_corruption(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);(root/'shard').write_bytes(b'ab')
            manifest=dict(repo='test',revision='frozen',files={},shards=[dict(name='shard',size=2,hub_lfs_sha256=hashlib.sha256(b'ab').hexdigest())])
            self.assertEqual(len(M.verify(root,manifest)['files']),1)
            (root/'shard').write_bytes(b'cd')
            with self.assertRaisesRegex(ValueError,'SHA_MISMATCH'):M.verify(root,manifest)
            (root/'shard').unlink()
            with self.assertRaisesRegex(ValueError,'MISSING_OR_SIZE'):M.verify(root,manifest)
