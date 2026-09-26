import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import analyze_round10_t2 as analysis


class T2AnalysisTests(unittest.TestCase):
    def test_nested_stage_intervals_are_validated(self):
        base = dict(request_id="r", generation=0, request_tokens=64,
                    batch_tokens=32, start=0, threshold=129,
                    path="HYBRID", basis="REQUEST_TOTAL")
        rows = [dict(base, stage="SELECTION")]
        for layer in (0, 3):
            for stage, start, end in (
                ("ORIGINAL_HYBRID_APPLY", 0, 100),
                ("CPU_SUBMIT", 10, 20),
                ("NPU_RESIDENT_APPLY", 30, 40),
                ("CPU_RESULT", 50, 90),
            ):
                rows.append(dict(base, layer=layer, stage=stage, outcome="RETURN",
                                 start_ns=start, end_ns=end))
        result = analysis.summarize(rows)
        self.assertEqual(result["selection_count"], 1)
        self.assertEqual(result["timed_layer_apply_count"], 2)

    def test_interval_outside_apply_is_rejected(self):
        base = dict(request_id="r", generation=0, request_tokens=64,
                    batch_tokens=32, start=0, threshold=129,
                    path="HYBRID", basis="REQUEST_TOTAL")
        rows = [dict(base, stage="SELECTION")]
        for stage, start, end in (
            ("ORIGINAL_HYBRID_APPLY", 10, 100),
            ("CPU_SUBMIT", 0, 20),
            ("NPU_RESIDENT_APPLY", 30, 40),
            ("CPU_RESULT", 50, 90),
        ):
            rows.append(dict(base, layer=0, stage=stage, outcome="RETURN",
                             start_ns=start, end_ns=end))
        with self.assertRaises(ValueError):
            analysis.summarize(rows)


if __name__ == "__main__":
    unittest.main()
