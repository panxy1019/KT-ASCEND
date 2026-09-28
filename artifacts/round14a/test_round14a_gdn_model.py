import json
import unittest
from pathlib import Path

from round14a_gdn_model_driver import ROLES, build_schedule
from round14a_gdn_branch_binding import EXPECTED_GDN_LAYERS, gdn_branch_model_ready


class GDNModelScheduleTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        root = Path(__file__).parent / "a-gdn-component-snapshots" / "v290"
        cls.fixture10 = json.loads((root / "round10-t1-fixture.json").read_text())
        cls.fixture11 = json.loads((root / "round11-t1-fixture.json").read_text())

    def test_two_smoke_identities_have_eight_attempts_and_one_abort(self):
        for version, implementation in ((292, "baseline"), (293, "candidate")):
            rows = build_schedule(self.fixture10, self.fixture11, version)
            self.assertEqual(ROLES[version], ("smoke", implementation, "S"))
            self.assertEqual(len(rows), 8)
            self.assertEqual(sum(row["phase"] == "cancel" for row in rows), 1)

    def test_abba_identities_have_twenty_attempts_and_no_abort(self):
        expected = {
            294: ("abba", "baseline", "A"),
            295: ("abba", "candidate", "A"),
            296: ("abba", "candidate", "B"),
            297: ("abba", "baseline", "B"),
        }
        for version, identity in expected.items():
            rows = build_schedule(self.fixture10, self.fixture11, version)
            self.assertEqual(ROLES[version], identity)
            self.assertEqual(len(rows), 20)
            self.assertFalse(any(row["phase"] == "cancel" for row in rows))
            for workload in ("W1", "W2", "W4"):
                selected = [row for row in rows if row.get("workload") == workload]
                self.assertEqual([row["phase"] for row in selected], [
                    "warmup", "measurement", "measurement", "measurement"
                ])

    def test_actual_branch_hook_binds_exact_gdn_topology_and_logs_once(self):
        import os
        import tempfile

        class Module:
            def __init__(self, layer):
                self.layer_id = layer
                self.round14_gdn_backend_enabled = True
                self.hook = None

            def register_forward_hook(self, hook):
                self.hook = hook
                return (self.layer_id, hook)

        modules = {layer: Module(layer) for layer in EXPECTED_GDN_LAYERS}

        class Model:
            def named_modules(self):
                for layer, module in modules.items():
                    module.__class__.__name__ = "Qwen4ReferenceGatedDeltaNet"
                    yield str(layer), module

        class Runner:
            model = Model()

        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "branches.jsonl"
            previous = os.environ.get("ROUND14A_GDN_BRANCH_PATH")
            os.environ["ROUND14A_GDN_BRANCH_PATH"] = str(path)
            try:
                runner = Runner()
                gdn_branch_model_ready(runner)
                for module in modules.values():
                    module.hook(module, (), object())
                    module.hook(module, (), object())
                rows = [json.loads(line) for line in path.read_text().splitlines()]
                self.assertEqual(len(rows), len(EXPECTED_GDN_LAYERS))
                self.assertEqual(sorted(row["layer"] for row in rows), list(EXPECTED_GDN_LAYERS))
                self.assertEqual({row["implementation"] for row in rows}, {"candidate"})
                self.assertTrue(all(row["completed_forward"] for row in rows))
            finally:
                if previous is None:
                    os.environ.pop("ROUND14A_GDN_BRANCH_PATH", None)
                else:
                    os.environ["ROUND14A_GDN_BRANCH_PATH"] = previous


if __name__ == "__main__":
    unittest.main()
