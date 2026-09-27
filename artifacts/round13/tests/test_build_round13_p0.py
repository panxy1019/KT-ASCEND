from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[3]
ROUND13 = ROOT / "artifacts" / "round13"
sys.path.insert(0, str(ROUND13))

from build_round13_p0 import build  # noqa: E402


class BuildRound13P0Test(unittest.TestCase):
    def test_build_freezes_parent_workloads_profiles_and_budget(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory) / "round13"
            build(ROOT, out)

            baseline = json.loads((out / "BASELINE_MANIFEST.json").read_text())
            self.assertEqual(baseline["status"], "ROUND13_P0_BASELINE_IDENTITY_FROZEN")
            self.assertEqual(
                baseline["model"]["revision"],
                "de4b8e4d43b917e7706784d8bb445c9af86a3540",
            )
            self.assertEqual(baseline["configuration"]["resident_budget_experts"], 1536)
            self.assertEqual(baseline["configuration"]["dynamic"], "OFF")
            self.assertEqual(baseline["configuration"]["deferred"], 0)
            self.assertTrue(baseline["parent_runs"]["v174"]["qualified"])
            self.assertTrue(baseline["parent_runs"]["v177"]["qualified"])

            workloads = json.loads((out / "WORKLOADS.json").read_text())
            expected = {
                "W0": (64, 2, 129),
                "W1": (128, 16, 129),
                "W2": (128, 128, 129),
                "W3": (512, 16, 513),
                "W4": (64, 32, 129),
            }
            for name, (input_tokens, output_tokens, threshold) in expected.items():
                row = workloads["workloads"][name]
                self.assertEqual(
                    (row["input_tokens"], row["output_tokens"], row["hybrid_threshold"]),
                    (input_tokens, output_tokens, threshold),
                )
            self.assertEqual(workloads["metrics"]["serial_output_tokens_per_second"]["scope"], "MEASUREMENT_WINDOW")
            self.assertFalse(workloads["metrics"]["tpot"]["network_batch_is_token_clock"])

            placements = json.loads((out / "PLACEMENT_PROFILES.json").read_text())
            self.assertEqual(placements["budget"]["experts"], 1536)
            self.assertEqual(placements["random"]["primary_seed"], 42)
            self.assertEqual(set(placements["profiles"]), {"uniform", "frequency", "front_loading", "random_seed42"})
            for row in placements["profiles"].values():
                self.assertEqual(row["budget_experts"], 1536)
            self.assertEqual(placements["profiles"]["front_loading"]["per_layer_counts"][:4], [512, 512, 512, 0])

            budget = json.loads((out / "ROUND13_EXECUTION_BUDGET.json").read_text())
            self.assertEqual(budget["limits"]["total"]["model_processes"], 28)
            self.assertEqual(budget["used"]["total"]["model_processes"], 0)
            self.assertEqual(budget["automatic_retries"], 0)

            resource = json.loads((out / "ROUND13_RESOURCE_CONTRACT.json").read_text())
            self.assertEqual(resource["device"]["physical"], 12)
            self.assertEqual(resource["cpu"]["numa_nodes"], [2, 3, 4, 7])
            self.assertEqual(resource["cpu"]["admission_rounds"], 2)
            self.assertEqual(resource["cpu"]["seconds_per_round"], 60)

            for name in (
                "round10-t1-fixture.json",
                "round11-t1-fixture.json",
                "qwen38_sched_dev.json",
                "qwen38_sched_holdout.json",
                "frequency-q32-source.json",
            ):
                self.assertTrue((out / "frozen-inputs" / name).is_file())

    def test_build_refuses_to_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory) / "round13"
            build(ROOT, out)
            with self.assertRaises(FileExistsError):
                build(ROOT, out)


if __name__ == "__main__":
    unittest.main()
