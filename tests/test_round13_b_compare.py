import importlib.util
from pathlib import Path
import unittest


COMPARATOR = (
    Path(__file__).parents[1]
    / "artifacts"
    / "round13"
    / "compare_round13_b.py"
)


def load_comparator():
    spec = importlib.util.spec_from_file_location("compare_round13_b", COMPARATOR)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run(implementation, block, *, w2_tpot):
    metrics = {}
    for workload in ("W0", "W1", "W2", "W3", "W4"):
        metrics[workload] = {
            "e2e_seconds_median": 1.0,
            "ttft_seconds_median": 1.0,
            "tpot_seconds_median": w2_tpot if workload == "W2" else 1.0,
            "serial_output_tokens_per_second": 1.0,
            "requests_per_second": 1.0,
        }
    return {
        "implementation": implementation,
        "block": block,
        "metrics": metrics,
        "qsa_complete": True,
        "ple_complete": True,
    }


class Round13BCompareTest(unittest.TestCase):
    def test_promotes_only_when_both_abba_w2_blocks_improve(self):
        self.assertTrue(COMPARATOR.exists())
        comparator = load_comparator()
        result = comparator.compare(
            [
                run("baseline", "A", w2_tpot=1.0),
                run("candidate", "A", w2_tpot=0.9),
                run("candidate", "B", w2_tpot=0.9),
                run("baseline", "B", w2_tpot=1.0),
            ],
            {"baseline": True, "candidate": True},
        )
        self.assertEqual(result["status"], "B_PROMOTED")
        self.assertEqual(result["primary_w2_tpot_ratios"], [0.9, 0.9])


if __name__ == "__main__":
    unittest.main()
