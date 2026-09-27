import importlib.util
import json
from pathlib import Path
import unittest


DRIVER = (
    Path(__file__).parents[1]
    / "artifacts"
    / "round13"
    / "round13_f_model_driver.py"
)
FIXTURE10 = (
    Path(__file__).parents[1]
    / "artifacts"
    / "round13"
    / "frozen-inputs"
    / "round10-t1-fixture.json"
)
FIXTURE11 = (
    Path(__file__).parents[1]
    / "artifacts"
    / "round13"
    / "frozen-inputs"
    / "round11-t1-fixture.json"
)


def load_driver():
    spec = importlib.util.spec_from_file_location("round13_f_model_driver", DRIVER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Round13FDriverTest(unittest.TestCase):
    def test_each_final_arm_has_twenty_performance_and_four_quality_requests(self):
        self.assertTrue(DRIVER.exists())
        driver = load_driver()
        fixtures = {
            "round10": json.loads(FIXTURE10.read_text()),
            "round11": json.loads(FIXTURE11.read_text()),
        }
        for version, (implementation, block) in driver.ROLES.items():
            performance = driver.performance_schedule(
                fixtures, version, implementation, block
            )
            quality = driver.quality_schedule(
                fixtures, version, implementation, block
            )
            self.assertEqual(len(performance), 20)
            self.assertEqual(len(quality), 4)
            self.assertEqual(
                sum(row["phase"] == "cancel" for row in quality),
                1 if block == "A" else 0,
            )


if __name__ == "__main__":
    unittest.main()
