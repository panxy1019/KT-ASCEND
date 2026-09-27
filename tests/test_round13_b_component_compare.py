import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


COMPARATOR = (
    Path(__file__).parents[1]
    / "artifacts"
    / "round13"
    / "compare_round13_b_component.py"
)


def load_comparator():
    spec = importlib.util.spec_from_file_location("compare_round13_b_component", COMPARATOR)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Round13BComponentCompareTest(unittest.TestCase):
    def test_summarize_accepts_exact_initialized_allocator_baseline(self):
        comparator = load_comparator()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "result.json"
            path.write_text(
                json.dumps(
                    {
                        "block": "A",
                        "status": "PASS_ROUND13_B_QSA_DECODE_COMPONENT",
                        "baseline_allocated": 16896,
                        "baseline_reserved": 2097152,
                        "after_allocated": 16896,
                        "after_reserved": 2097152,
                        "rows": [],
                        "contract_checks": {},
                    }
                )
            )
            self.assertTrue(comparator.summarize(path)["allocator_cleanup_pass"])

    def test_two_complete_blocks_below_point_nine_pass(self):
        self.assertTrue(COMPARATOR.exists())
        comparator = load_comparator()
        rows = [
            {
                "block": block,
                "status": "PASS_ROUND13_B_QSA_DECODE_COMPONENT",
                "row_count": 9,
                "ratios": [0.5] * 9,
                "numerical_pass": True,
                "contract_checks_pass": True,
                "allocator_cleanup_pass": True,
            }
            for block in ("A", "B")
        ]
        result = comparator.compare(rows)
        self.assertEqual(result["status"], "PASS_ROUND13_B_COMPONENT_GATE")


if __name__ == "__main__":
    unittest.main()
