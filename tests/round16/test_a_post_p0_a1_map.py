from __future__ import annotations

import json
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[2]
MAP_PATH = ROOT / "artifacts/round16-a/POST_P0_A1_MAP.json"


class PostP0A1MapTests(unittest.TestCase):
    def setUp(self) -> None:
        self.document = json.loads(MAP_PATH.read_text())

    def test_map_is_offline_only_default_off_and_names_real_start_chain(self) -> None:
        self.assertEqual(self.document["status"], "READY_OFFLINE_MAP_DEVICE_REQUIRED_NOT_RUN")
        self.assertFalse(self.document["default_enabled"])
        self.assertEqual(self.document["execution"]["model_processes_started"], 0)
        self.assertEqual(self.document["execution"]["device_runs"], "NOT_RUN")

        startup = self.document["p0"]["s0_startup"]
        self.assertEqual(startup["host_controller"]["symbol"], "main")
        self.assertEqual(startup["host_launcher"]["symbol"], "<module>")
        self.assertIn(
            "/out/source/runtime/candidate_server.py",
            startup["container_process"],
        )
        self.assertEqual(startup["request_driver"]["symbol"], "run")
        self.assertEqual(
            startup["chain"],
            [
                "round15_p0_controller.main",
                "source/tools/launch_hybrid.py",
                "/out/source/runtime/candidate_server.py",
                "round15_p0_supervisor.main",
                "round15_p0_driver.run",
            ],
        )

    def test_profile_blocks_clocks_and_recording_points_are_bounded(self) -> None:
        profile = self.document["p0"]["profile"]
        self.assertEqual(profile["schedule"], ["WARMUP", "W1", "W2", "DRAIN"])
        self.assertEqual([block["block"] for block in profile["accepted_blocks"]], ["A", "B"])
        self.assertEqual([block["version"] for block in profile["accepted_blocks"]], [335, 338])
        self.assertTrue(profile["accepted_blocks"][0]["salvaged_raw"])
        self.assertFalse(profile["accepted_blocks"][1]["salvaged_raw"])
        self.assertEqual(
            {clock["domain"] for clock in self.document["p0"]["clock_domains"]},
            {"HOST_MONOTONIC_NS", "NPU_EVENT_ELAPSED_MS"},
        )

        points = self.document["p0"]["recording_points"]
        self.assertEqual(points["CPU_EXPERT"]["status"], "MISSING_DIRECT_MEASUREMENT")
        self.assertEqual(points["NPU"]["component"], "NPU_RESIDENT")
        self.assertEqual(points["HOST_DISPATCH"]["component"], "CPU_SUBMIT")
        self.assertEqual(points["HOST_SYNC"]["component"], "CPU_RESULT_WAIT")
        self.assertEqual(points["UNKNOWN"]["derivation"], "NEVER_PARENT_MINUS_CHILDREN")

    def test_budget_and_candidate_gate_fail_closed_without_new_device_evidence(self) -> None:
        budget = self.document["p0"]["budget"]
        self.assertEqual(
            budget["two_fresh_blocks"],
            {
                "model_processes": 2,
                "generation_attempts": 8,
                "abort_controls": 0,
                "automatic_retries": 0,
            },
        )
        self.assertEqual(budget["per_block_generation_attempts"], 4)

        gate = self.document["a1"]["candidate_gate"]
        self.assertEqual(gate["p0_order"], ["QSA_INDEXER", "GDN_RMSNORM_IF_A1_INAPPLICABLE"])
        self.assertEqual(gate["request_gate"], "DECODE_W2")
        self.assertEqual(gate["round15_component_gate"], "PASS")
        self.assertEqual(gate["round15_request_gate"], "FAIL_NO_W2_GAIN")
        self.assertEqual(gate["round16_selected_candidate"], None)
        self.assertFalse(gate["enabled"])

        missing = self.document["missing_evidence"]
        self.assertTrue(any(item["field"] == "CPU_EXPERT.start_ns/end_ns" for item in missing))
        self.assertTrue(any(item["field"] == "span.clock_domain" for item in missing))
        self.assertTrue(any(item["field"] == "Round16 fresh block A/B final results" for item in missing))


if __name__ == "__main__":
    unittest.main()
