"""Synthetic CPU-only adapter integration. Never a model qualification result."""
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from evaluate_stateful_contract import canonical, main
from test_stateful_contract_gate import fixture

try:
    import torch
except ImportError:
    torch = None


@unittest.skipIf(torch is None, 'torch unavailable locally')
class SyntheticEvaluatorIntegration(unittest.TestCase):
    def test_identity_fixture_derives_zero_reserves_after_exposure_gate(self):
        import evaluate_contract_v2 as metrics
        import tail_statistics as stats
        from assemble_capture import byte_sha
        torch.set_num_threads(1)
        corpus, prereg, manifests = fixture(8)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            logits = torch.zeros(1, 102400, dtype=torch.bfloat16)
            logits[0, 2] = 1
            capture = root / 'synthetic.pt'
            torch.save({'capture.next_token_logits': logits}, capture)
            for manifest in manifests:
                for row in manifest['rows']:
                    row.update(capture_file=str(capture), logits_sha256=byte_sha(logits[0]), serving_token=2)
            prereg.update(metrics_module_sha256=hashlib.sha256(Path(metrics.__file__).read_bytes()).hexdigest(),
                          statistics_module_sha256=hashlib.sha256(Path(stats.__file__).read_bytes()).hexdigest())
            for name, value in [('corpus', corpus), ('prereg', prereg), ('base', manifests[0]), ('defer', manifests[1])]:
                value = {k:v for k,v in value.items() if k != 'sha256'}
                value['sha256'] = canonical(value)
                (root / (name + '.json')).write_text(json.dumps(value))
            argv = ['evaluate', 'derive', '--baseline', str(root/'base.json'), '--deferred', str(root/'defer.json'),
                    '--corpus', str(root/'corpus.json'), '--preregistration', str(root/'prereg.json'), '--output', str(root/'out')]
            with patch.object(sys, 'argv', argv):
                with self.assertRaises(SystemExit) as exc:
                    main()
            self.assertEqual(exc.exception.code, 0)
            candidate = json.loads((root/'out/candidate.json').read_text())
            self.assertEqual({r['threshold'] for r in candidate['reserves'].values()}, {0.})
            self.assertEqual({r['rank_1based'] for r in candidate['reserves'].values()}, {127})


if __name__ == '__main__':
    unittest.main()
