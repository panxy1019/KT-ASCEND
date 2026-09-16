import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

TOOLS=Path(__file__).parent


class EntryGuardTests(unittest.TestCase):
    def test_quality_cannot_start_from_unapproved_contract(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);approval=root/'approval.json'
            approval.write_text(json.dumps(dict(quality_execution_allowed=False)))
            output=root/'output'
            run=subprocess.run([sys.executable,str(TOOLS/'run_quality_screen.py'),
                '--approval',str(approval),'--inputs',str(root/'missing-input'),
                '--profile',str(root/'missing-profile'),'--parser-dir',str(root),
                '--output',str(output)],capture_output=True,text=True)
            self.assertNotEqual(run.returncode,0)
            self.assertIn('HELDOUT_REVIEW_REQUIRED',run.stderr)
            self.assertFalse(output.exists())

    def test_heldout_rejects_candidate_hash_drift_before_runtime_access(self):
        with tempfile.TemporaryDirectory() as d:
            candidate=Path(d)/'candidate.json';candidate.write_text('{}')
            run=subprocess.run([sys.executable,str(TOOLS/'run_hd6_capture.py'),
                '--candidate',str(candidate),'--candidate-sha256','invalid'],capture_output=True,text=True)
            self.assertNotEqual(run.returncode,0)
            self.assertIn('AssertionError',run.stderr)
            self.assertNotIn('FileNotFoundError',run.stderr)


if __name__=='__main__':unittest.main()
