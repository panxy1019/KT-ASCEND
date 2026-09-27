import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


RUNNER = Path(__file__).parents[1] / "artifacts" / "round13" / "round13_f_model_runner.py"


def load_runner():
    spec = importlib.util.spec_from_file_location("round13_f_model_runner", RUNNER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Round13FRunnerTest(unittest.TestCase):
    def test_final_result_requires_24_requests_and_cleanup(self):
        self.assertTrue(RUNNER.exists())
        runner = load_runner()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            final = {
                "status": "PASS_ROUND13_F_MODEL_PROCESS",
                "version": 227,
                "implementation": "baseline",
                "block": "A",
                "generation_attempts": 24,
                "abort_controls": 1,
                "automatic_retries": 0,
                "measurement_windows": {name: {} for name in ("W0", "W1", "W2", "W3", "W4")},
            }
            supervisor = {
                "status": "PASS_ROUND13_F_MODEL_PROCESS",
                "guard_exit_ack": {"action": "OWNED_EXITED"},
                "owned_container_removed": True,
                "physical12_after": "EMPTY",
            }
            (root / "round13-f-final.json").write_text(json.dumps(final))
            (root / "round13-model-supervisor.json").write_text(json.dumps(supervisor))
            runner.strict_result(root, 227, "baseline", "A")


if __name__ == "__main__":
    unittest.main()
