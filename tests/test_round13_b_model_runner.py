import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


RUNNER = (
    Path(__file__).parents[1]
    / "artifacts"
    / "round13"
    / "round13_b_model_runner.py"
)


def load_runner():
    spec = importlib.util.spec_from_file_location("round13_b_model_runner", RUNNER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Round13BModelRunnerTest(unittest.TestCase):
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

    def test_smoke_result_requires_complete_cleanup_and_budget(self):
        self.assertTrue(RUNNER.exists())
        runner = load_runner()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            final = {
                "status": "PASS_ROUND13_B_SMOKE_PROCESS",
                "version": 217,
                "implementation": "baseline",
                "mode": "smoke",
                "block": "A",
                "generation_attempts": 8,
                "abort_controls": 1,
                "automatic_retries": 0,
                "measurement_windows": {},
            }
            supervisor = {
                "status": "PASS_ROUND13_B_SMOKE_PROCESS",
                "guard_exit_ack": {"action": "OWNED_EXITED"},
                "owned_container_removed": True,
                "physical12_after": "EMPTY",
            }
            (root / "round13-b-final.json").write_text(json.dumps(final))
            (root / "round13-model-supervisor.json").write_text(json.dumps(supervisor))

            result = runner.strict_result(
                root, 217, "baseline", "smoke", "A"
            )

            self.assertEqual(result["generation_attempts"], 8)

    def test_request_result_requires_all_five_measurement_windows(self):
        self.assertTrue(RUNNER.exists())
        runner = load_runner()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            final = {
                "status": "PASS_ROUND13_B_REQUEST_PROCESS",
                "version": 219,
                "implementation": "baseline",
                "mode": "request",
                "block": "A",
                "generation_attempts": 20,
                "abort_controls": 0,
                "automatic_retries": 0,
                "measurement_windows": {name: {} for name in ("W0", "W1", "W2", "W3", "W4")},
            }
            supervisor = {
                "status": "PASS_ROUND13_B_REQUEST_PROCESS",
                "guard_exit_ack": {"action": "OWNED_EXITED"},
                "owned_container_removed": True,
                "physical12_after": "EMPTY",
            }
            (root / "round13-b-final.json").write_text(json.dumps(final))
            (root / "round13-model-supervisor.json").write_text(json.dumps(supervisor))

            runner.strict_result(root, 219, "baseline", "request", "A")
            final["measurement_windows"].pop("W4")
            (root / "round13-b-final.json").write_text(json.dumps(final))
            with self.assertRaisesRegex(RuntimeError, "strict process result"):
                runner.strict_result(root, 219, "baseline", "request", "A")


if __name__ == "__main__":
    unittest.main()
