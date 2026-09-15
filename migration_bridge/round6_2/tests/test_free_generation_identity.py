from pathlib import Path
import sys
import unittest
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from run_free_generation_captured import expected_inputs


class FreeGenerationIdentityTests(unittest.TestCase):
    def test_prefill(self):
        self.assertEqual(expected_inputs([10, 20], [30, 40], 0), ([10, 20], [0, 1]))

    def test_decode(self):
        self.assertEqual(expected_inputs([10, 20], [30, 40], 1), ([30], [2]))

    def test_lookahead(self):
        self.assertEqual(expected_inputs([10, 20], [30, 40], 2), ([40], [3]))
