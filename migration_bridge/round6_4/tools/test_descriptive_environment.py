import unittest
from run_descriptive_holdout import capture_environment


class CaptureEnvironment(unittest.TestCase):
    def test_preserves_inherited_environment(self):
        original = {'PYTHONPATH':'/cann:/sglang', 'UNCHANGED':'value'}
        result = capture_environment(original)
        self.assertEqual(result['PYTHONPATH'],
            '/workspace/results/round6_2-prequalification/tools:/cann:/sglang')
        self.assertEqual(result['UNCHANGED'],'value')
        self.assertEqual(original['PYTHONPATH'],'/cann:/sglang')

    def test_empty_pythonpath(self):
        self.assertTrue(capture_environment({})['PYTHONPATH'].startswith(
            '/workspace/results/round6_2-prequalification/tools'))


if __name__ == '__main__':
    unittest.main()
