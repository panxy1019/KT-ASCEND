from pathlib import Path
import sys
import unittest
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from run_runtime_determinism import validate


class RuntimeValidationTests(unittest.TestCase):
    def answer(self):
        return dict(output_ids=[1, 2], meta_info=dict(id="r", output_token_logprobs=[[-.2, 1], [-.1, 2]]))

    def test_valid(self):
        self.assertEqual(validate(self.answer(), "r", 2), [1, 2])

    def test_wrong_request(self):
        with self.assertRaises(ValueError):
            validate(self.answer(), "other", 2)

    def test_short_response(self):
        with self.assertRaises(ValueError):
            validate(self.answer(), "r", 64)

    def test_nonfinite(self):
        answer = self.answer()
        answer["meta_info"]["output_token_logprobs"][0][0] = float("nan")
        with self.assertRaises(ValueError):
            validate(answer, "r", 2)

    def test_logprob_token_mismatch(self):
        answer = self.answer()
        answer["meta_info"]["output_token_logprobs"][0][1] = 4
        with self.assertRaises(ValueError):
            validate(answer, "r", 2)
