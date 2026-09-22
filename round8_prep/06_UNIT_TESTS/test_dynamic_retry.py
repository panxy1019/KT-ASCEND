import copy
import json
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))


class RetryTest(unittest.TestCase):
    def test_only_timeout_and_output_identity_change(self):
        from run_qwen_dynamic_heldout_retry import retry_config
        original = json.loads((ROOT / '11_DYNAMIC/qwen-dynamic-candidate-v2/heldout.json').read_text())
        before = copy.deepcopy(original)
        updated = retry_config(original)
        self.assertEqual(original, before)
        self.assertEqual(updated['argv'][-2:], ['--watchdog-timeout', '1800'])
        updated['argv'] = updated['argv'][:-2]
        restored = json.loads(json.dumps(updated).replace('qwen-dynamic-heldout-v3', 'qwen-dynamic-heldout-v2'))
        self.assertEqual(restored, original)

    def test_existing_watchdog_override_rejected(self):
        from run_qwen_dynamic_heldout_retry import retry_config
        with self.assertRaises(AssertionError):
            retry_config({'argv': ['python', '--watchdog-timeout', '9999']})

