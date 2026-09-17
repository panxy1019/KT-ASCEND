import json
from pathlib import Path
import tempfile
import unittest
from threshold_sweep import probe_summaries


class ProbePrefixTests(unittest.TestCase):
    def test_both_explicit_layouts(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            for prefix in ('threshold-','dynamic-dev-probe-'):
                for threshold in (256,512,1024):
                    folder=root/f'{prefix}{threshold}';folder.mkdir()
                    (folder/'summary.json').write_text(json.dumps({'status':'PASS'}))
                self.assertEqual(len(probe_summaries(root,prefix)),3)
    def test_failed_predecessor_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);folder=root/'threshold-256';folder.mkdir()
            (folder/'summary.json').write_text('{"status":"FAIL"}')
            with self.assertRaisesRegex(AssertionError,'TRIGGER_GENERATION_PREDECESSOR'):
                probe_summaries(root,'threshold-')
    def test_unknown_layout_rejected(self):
        with self.assertRaisesRegex(ValueError,'UNSUPPORTED_PROBE_PREFIX'):
            probe_summaries(Path('/unused'),'../')
