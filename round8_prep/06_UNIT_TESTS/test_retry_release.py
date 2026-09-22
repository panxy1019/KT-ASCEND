import sys
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))


class RetryReleaseTest(unittest.TestCase):
    def test_release_cannot_relabel_failed_v2_as_pass(self):
        from qwen_dynamic_retry_release import validate_release_identity
        good=dict(heldout='PASS',cmmlu='PASS',heldout_attempt='qwen-dynamic-heldout-v3',retry_sha256='a'*64)
        validate_release_identity(good,'a'*64)
        for bad in [dict(good,heldout_attempt='qwen-dynamic-heldout-v2'),
                    dict(good,heldout='RUNNING'),dict(good,cmmlu='FAIL'),
                    dict(good,retry_sha256='b'*64)]:
            with self.assertRaises(AssertionError):validate_release_identity(bad,'a'*64)
