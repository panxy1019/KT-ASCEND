import hashlib
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


ROOT=Path(__file__).resolve().parents[2]


class FailureFreezeTests(unittest.TestCase):
    def test_reproduces_all_discordances_exactly(self):
        with tempfile.TemporaryDirectory() as d:
            output=Path(d)/'fixture.json'
            subprocess.run([sys.executable,str(Path(__file__).with_name('freeze_quality_failure.py')),
                '--quality',str(ROOT/'round6_6/corpora/frozen-v1/quality.json'),
                '--pairs',str(ROOT/'round6_6/evidence/quality_v1/postprocess/all_quality_pairs.json'),
                '--output',str(output)],check=True,capture_output=True)
            self.assertEqual(hashlib.sha256(output.read_bytes()).hexdigest(),
                             '81e78b3ce6902d49daf6b0c8558424ce4d9eca40d5fbed41c7376ad3fe29d4e1')

    def test_rejects_modified_pair_evidence(self):
        with tempfile.TemporaryDirectory() as d:
            fake=Path(d)/'pairs.json';fake.write_text('{}')
            output=Path(d)/'fixture.json'
            result=subprocess.run([sys.executable,str(Path(__file__).with_name('freeze_quality_failure.py')),
                '--quality',str(ROOT/'round6_6/corpora/frozen-v1/quality.json'),
                '--pairs',str(fake),'--output',str(output)],capture_output=True)
            self.assertNotEqual(result.returncode,0)
            self.assertIn(b'AssertionError',result.stderr)
            self.assertFalse(output.exists())


if __name__=='__main__':unittest.main()
