import importlib.util
from pathlib import Path
import unittest


COMPARATOR = Path(__file__).parents[1] / "artifacts" / "round13" / "compare_round13_f.py"


def load_comparator():
    spec = importlib.util.spec_from_file_location("compare_round13_f", COMPARATOR)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run(implementation, block, tpot):
    metrics = {
        workload: {
            "e2e_seconds_median": 1.0,
            "ttft_seconds_median": 1.0,
            "tpot_seconds_median": tpot if workload == "W2" else 1.0,
            "serial_output_tokens_per_second": 1.0,
            "requests_per_second": 1.0,
        }
        for workload in ("W0", "W1", "W2", "W3", "W4")
    }
    return {
        "implementation": implementation,
        "block": block,
        "metrics": metrics,
        "quality_complete": True,
        "qsa_complete": True,
        "ple_complete": True,
    }


class Round13FCompareTest(unittest.TestCase):
    def test_promotes_final_only_when_both_pairs_pass(self):
        self.assertTrue(COMPARATOR.exists())
        comparator = load_comparator()
        result = comparator.compare([
            run("baseline", "A", 1.0),
            run("candidate", "A", 0.9),
            run("candidate", "B", 0.9),
            run("baseline", "B", 1.0),
        ])
        self.assertEqual(result["status"], "F_FINAL_PROMOTED")


if __name__ == "__main__":
    unittest.main()
