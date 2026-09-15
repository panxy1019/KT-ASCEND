from pathlib import Path
import sys
import unittest
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from run_quality_checked import validate_response
from compare_quality_checked import paired_summary, percentile
import random


def response(score=-.5):
    return dict(meta_info=dict(input_token_logprobs=[[None, 1], [score, 2]],
                               output_token_logprobs=[[-.2, 3]]))


class QualitySanityTests(unittest.TestCase):
    def test_zero_discordance(self):
        report = paired_summary([True, False], [True, False], random.Random(6202))
        self.assertEqual(report["paired_bootstrap_ci95"], [0, 0])
        self.assertEqual(report["discordant_correctness_pairs"], 0)

    def test_complete_decline(self):
        report = paired_summary([True] * 128, [False] * 128, random.Random(6202))
        self.assertEqual(report["status"], "QUALITY_REGRESSION")
        self.assertEqual(report["paired_bootstrap_ci95"], [-1, -1])

    def test_percentile_interpolation(self):
        self.assertEqual(percentile([0, 1, 2, 3], .5), 1.5)

    def test_valid(self):
        item = response()
        self.assertIs(validate_response(item), item)

    def test_nonfinite(self):
        for value in (float("nan"), float("inf"), None):
            with self.assertRaises(ValueError):
                validate_response(response(value))

    def test_missing_output(self):
        item = response()
        item["meta_info"]["output_token_logprobs"] = []
        with self.assertRaises(ValueError):
            validate_response(item)

    def test_invalid_token(self):
        item = response()
        item["meta_info"]["input_token_logprobs"][-1][1] = -1
        with self.assertRaises(ValueError):
            validate_response(item)
