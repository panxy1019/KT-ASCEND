import copy
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from run_qwen_dynamic_perf_retry import retry_config


class PerfRetryTest(unittest.TestCase):
    def test_long_forward_survives_bounded_watchdog_without_changing_work(self):
        original = {'argv': ['python', '-m', 'serve', '--kt-gpu-prefill-token-threshold', '256'],
                    'env': {'CPU_THREADS': '64'}, 'code_hashes': {'x': 'abc'}}
        before = copy.deepcopy(original)
        actual = retry_config(original)
        self.assertEqual(actual['argv'][-2:], ['--watchdog-timeout', '1800'])
        self.assertEqual(actual['argv'][:-2], before['argv'])
        self.assertEqual(actual['env'], before['env'])
        self.assertEqual(actual['code_hashes'], before['code_hashes'])
        self.assertEqual(original, before)

    def test_conflicting_timeout_is_rejected(self):
        with self.assertRaises(AssertionError):
            retry_config({'argv': ['serve', '--watchdog-timeout', '300']})


if __name__ == '__main__':
    unittest.main()
