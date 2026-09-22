import json
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile
import unittest


class FinalAudit(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.base = Path(cls.temp.name)
        here = Path(__file__).resolve()
        cls.tool = here.with_name('audit_final_descriptive.py')
        archive = here.parents[1]/'evidence/final-descriptive-910b/metadata.tar.gz'
        with tarfile.open(archive) as stream:
            stream.extractall(cls.base, filter='data')
        cls.root = cls.base/'r64-910b-descriptive32-v1'

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def invoke(self, name):
        return subprocess.run([sys.executable,str(self.tool),'--root',str(self.root),
                               '--output',str(self.base/name)],capture_output=True,text=True)

    def test_archived_complete_cohort(self):
        result = self.invoke('pass.json')
        self.assertEqual(result.returncode,0,result.stderr)
        report = json.loads((self.base/'pass.json').read_text())
        self.assertEqual(report['physical_forwards'],26520)
        self.assertFalse(report['quality_or_speedup_gate'])

    def test_unknown_forward_rejected(self):
        path = self.root/'on_native/final-integrity.json'
        original = path.read_bytes()
        try:
            value = json.loads(original)
            value['unknown'] = 1
            path.write_text(json.dumps(value))
            self.assertNotEqual(self.invoke('reject.json').returncode,0)
        finally:
            path.write_bytes(original)


if __name__ == '__main__':
    unittest.main()
