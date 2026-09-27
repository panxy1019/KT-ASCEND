import importlib.util
from pathlib import Path
import unittest


COMPARATOR = (
    Path(__file__).parents[1]
    / "artifacts"
    / "round13"
    / "compare_round13_a.py"
)


def load_comparator():
    spec = importlib.util.spec_from_file_location("compare_round13_a", COMPARATOR)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run(strategy, tpot):
    metrics = {}
    for workload in ("W0", "W1", "W2", "W3", "W4"):
        metrics[workload] = {
            "e2e_seconds_median": 1.0,
            "ttft_seconds_median": 1.0,
            "tpot_seconds_median": tpot if workload == "W2" else 1.0,
            "serial_output_tokens_per_second": 1.0,
            "requests_per_second": 1.0,
        }
    return {
        "version": 1 if strategy == "uniform" else 2,
        "strategy": strategy,
        "block": "A",
        "metrics": metrics,
    }


class Round13ACompareTest(unittest.TestCase):
    def test_incomplete_pair_keeps_observed_first_block_ratio(self):
        comparator = load_comparator()
        result = comparator.compare(
            [run("uniform", 1.0), run("frequency", 0.9)]
        )
        frequency = result["candidates"]["frequency"]
        self.assertEqual(frequency["status"], "INCOMPLETE_PAIR")
        self.assertEqual(
            frequency["observed_first_block_ratios"]["W2"]["tpot"], 0.9
        )


if __name__ == "__main__":
    unittest.main()
