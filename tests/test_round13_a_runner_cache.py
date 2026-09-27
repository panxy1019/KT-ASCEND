import importlib.util
from pathlib import Path
import unittest


RUNNER = (
    Path(__file__).parents[1]
    / "artifacts"
    / "round13"
    / "round13_a_arm_runner_v8.py"
)


def load_runner():
    spec = importlib.util.spec_from_file_location("round13_a_arm_runner_v8", RUNNER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Round13ACacheReceiptTest(unittest.TestCase):
    def test_zero_residency_is_a_valid_audited_teardown(self):
        runner = load_runner()
        self.assertTrue(
            runner.valid_cache_retirement(
                {
                    "status": "NOOP_ZERO_RESIDENCY",
                    "cache_retirement_performed": False,
                    "audit_complete": True,
                    "active_references": [],
                    "stat_identity_unchanged": True,
                }
            )
        )

    def test_active_reference_never_passes(self):
        runner = load_runner()
        self.assertFalse(
            runner.valid_cache_retirement(
                {
                    "status": "PASS_CACHE_RETIRED",
                    "cache_retirement_performed": True,
                    "audit_complete": True,
                    "active_references": ["still-open"],
                    "stat_identity_unchanged": True,
                }
            )
        )


if __name__ == "__main__":
    unittest.main()
