import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]


def load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class Round12aModelBuilderTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.builder = load("round12a_model_builder")

    def test_candidate_bank_nonkt_branch_precedes_bank_receipt(self):
        source = (ROOT / "fixtures/v154/candidate_bank.py").read_text()
        patched = self.builder.patch_candidate_bank(source)
        branch = patched.index("ROUND12A_NONKT_MODE")
        receipt = patched.index("ROUND9_BANK_RECEIPT")
        self.assertLess(branch, receipt)
        self.assertEqual(patched.count("bind_round12a_nonkt_source_identity"), 2)

    def test_nonkt_keeps_exact_promoted_qsa_tuple(self):
        self.assertEqual(
            self.builder.PROMOTED_QSA_SHA256,
            "0fd8d98b37f7e90e90f9866356892236dcc7d0114351d1438025561df6c1bd35",
        )
        self.assertEqual(
            self.builder.PROMOTED_QWEN_SHA256,
            "d4583c5abe9791b09c48d1dd8f8491a2988eb5147d45df188e0e48908c699f1f",
        )
        self.assertEqual(
            self.builder.PROMOTED_CONTEXT_BINDING_SHA256,
            "97eb1b60d0ee24305ad1a052884b79a3045fa3574fcfc85e8b2c4ccfdbfc0c11",
        )

    def test_roles_are_fixed_smoke_then_abba(self):
        self.assertEqual(
            self.builder.ROLES,
            {
                173: ("NON_KT", "SMOKE"),
                174: ("KT", "A"),
                175: ("NON_KT", "A"),
                176: ("NON_KT", "B"),
                177: ("KT", "B"),
            },
        )


class Round12aDriverTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.driver = load("round12a_model_driver")

    def test_smoke_is_one_frozen_request(self):
        fixture = {"cases": {"edge64": {"input_ids": list(range(64))}}}
        rows = self.driver.build_schedule(
            fixture, version=173, mode="NON_KT", block="SMOKE", workloads=["W1"]
        )
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["phase"], "smoke")
        self.assertEqual(rows[0]["decode_tokens"], 4)

    def test_abba_schedule_is_warmup_plus_three_measurements(self):
        fixture = {
            "cases": {
                "edge64": {"input_ids": list(range(64))},
                "edge128": {"input_ids": list(range(128))},
            }
        }
        rows = self.driver.build_schedule(
            fixture, version=174, mode="KT", block="A", workloads=["W1"]
        )
        self.assertEqual(len(rows), 4)
        self.assertEqual(
            [r["phase"] for r in rows],
            ["warmup", "measurement", "measurement", "measurement"],
        )
        self.assertTrue(all(r["decode_tokens"] == 2 for r in rows))

    def test_queue_decision_uses_observed_ttft_plus_one_decode_interval(self):
        self.assertEqual(
            self.driver.decide_workloads(
                smoke_ttft_seconds=815.045499455,
                smoke_tpot_seconds=389.66712424733333,
                smoke_e2e_seconds=1984.047039175,
            ),
            {
                "workloads": ["W1"],
                "projection_formula": "smoke_ttft_seconds + smoke_tpot_seconds",
                "projected_w1_seconds": 1204.7126237023333,
                "request_wall_limit_seconds": 1800,
                "source_smoke_exceeded_request_wall_limit": True,
            },
        )

    def test_frozen_abba_queue_requires_the_two_token_definition(self):
        valid = {
            "schema": 1,
            "status": "FROZEN_ROUND12A_ABBA_QUEUE",
            "source_smoke_version": 173,
            "workloads": ["W1"],
            "workload_definitions": {
                "W1": {"case_id": "edge64", "input_tokens": 64, "decode_tokens": 2}
            },
            "request_wall_limit_seconds": 1800,
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "queue.json"
            path.write_text(json.dumps(valid))
            workloads, digest = self.driver._load_queue(path, version=174)
            self.assertEqual(workloads, ["W1"])
            self.assertEqual(len(digest), 64)
            valid["workload_definitions"]["W1"]["decode_tokens"] = 4
            path.write_text(json.dumps(valid))
            with self.assertRaisesRegex(ValueError, "invalid Round12a ABBA queue"):
                self.driver._load_queue(path, version=174)

    def test_request_wall_gate_is_strict(self):
        self.driver.validate_request_wall(1799.999, version=174)
        with self.assertRaisesRegex(RuntimeError, "wall-time limit"):
            self.driver.validate_request_wall(1800.0, version=174)
        # The immutable v173 smoke predates the shortened ABBA workload and is
        # retained as correctness/cost evidence even though it exceeded 1800 s.
        self.driver.validate_request_wall(1984.047039175, version=173)


class Round12aIdentityTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.identity = load("round12a_nonkt_identity")

    def test_cached_binding_returns_qualified_receipt(self):
        receipt = {
            "schema": 1,
            "status": "PASS_ROUND12A_NONKT_OFFLOADER_BINDING",
            "source_identity": {"revision": "frozen"},
            "cpu_parameter_bytes": 123,
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "binding.json"
            path.write_text(json.dumps(receipt))
            actual = self.identity.bind_round12a_nonkt_identity(
                object(), receipt_path=path
            )
        self.assertEqual(actual, receipt)


if __name__ == "__main__":
    unittest.main()
