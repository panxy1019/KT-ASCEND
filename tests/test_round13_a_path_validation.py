import importlib.util
from pathlib import Path
import unittest


DRIVER = (
    Path(__file__).parents[1]
    / "artifacts"
    / "round13"
    / "a-model-transfer-v6"
    / "round13_a_model_driver.py"
)


def load_driver():
    spec = importlib.util.spec_from_file_location("round13_a_model_driver_v6", DRIVER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def front_profile():
    return {"strategy": "front_loading", "per_layer_counts": [512, 512, 512] + [0] * 45}


class Round13PathValidationTest(unittest.TestCase):
    def test_front_loading_accepts_cross_layer_hybrid_evidence(self):
        driver = load_driver()
        self.assertTrue(hasattr(driver, "validate_round13_hybrid_path"))
        events = [
            {"request_id": "r", "stage": "SELECTION", "path": "HYBRID", "basis": "REQUEST_TOTAL"},
            {"request_id": "r", "stage": "CPU_SUBMIT", "layer": 3, "outcome": "RETURN"},
            {"request_id": "r", "stage": "CPU_RESULT", "layer": 3, "outcome": "RETURN"},
            {"request_id": "r", "stage": "ORIGINAL_HYBRID_APPLY", "layer": 3, "outcome": "RETURN"},
        ]

        result = driver.validate_round13_hybrid_path("r", events, [], front_profile())

        self.assertEqual(result["cpu_submit_count"], 1)
        self.assertEqual(result["cpu_result_count"], 1)
        self.assertEqual(result["npu_evidence"], "FROZEN_FULL_RESIDENT_LAYER")

    def test_front_loading_still_requires_observed_cpu_completion(self):
        driver = load_driver()
        events = [
            {"request_id": "r", "stage": "SELECTION", "path": "HYBRID", "basis": "REQUEST_TOTAL"},
        ]

        with self.assertRaisesRegex(RuntimeError, "complete Hybrid path"):
            driver.validate_round13_hybrid_path("r", events, [], front_profile())


if __name__ == "__main__":
    unittest.main()
