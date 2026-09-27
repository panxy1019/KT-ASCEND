import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
MODULE = ROOT / "artifacts/round14/build_round14_a_snapshot.py"


def load_module():
    spec = importlib.util.spec_from_file_location("build_round14_a_snapshot", MODULE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Round14ASnapshotBuilderTest(unittest.TestCase):
    def test_contracts_freeze_k_n_n_k_and_budget(self):
        api = load_module()
        expected = {
            235: ("KT", "A", "kt-round14-a-k-a-v235"),
            236: ("NATIVE_TP8", "A", "kt-round14-a-n-a-v236"),
            237: ("NATIVE_TP8", "B", "kt-round14-a-n-b-v237"),
            238: ("KT", "B", "kt-round14-a-k-b-v238"),
            239: ("NATIVE_TP8", "A", "kt-round14-a-n-a-v239"),
            240: ("NATIVE_TP8", "B", "kt-round14-a-n-b-v240"),
            241: ("NATIVE_TP8", "A", "kt-round14-a-n-a-v241"),
            242: ("NATIVE_TP8", "B", "kt-round14-a-n-b-v242"),
            247: ("NATIVE_TP8", "A", "kt-round14-a-n-a-v247"),
            248: ("NATIVE_TP8", "B", "kt-round14-a-n-b-v248"),
            251: ("NATIVE_TP8", "A", "kt-round14-a-n-a-v251"),
            252: ("NATIVE_TP8", "B", "kt-round14-a-n-b-v252"),
        }
        for version, (backend, block, container) in expected.items():
            contract = api.contract(version)
            self.assertEqual(contract["backend"], backend)
            self.assertEqual(contract["block"], block)
            self.assertEqual(contract["container"], container)
            self.assertEqual(contract["generation_limit"], 24)
            self.assertEqual(contract["abort_limit"], 0)
            self.assertEqual(contract["profiler"], "OFF")
            self.assertEqual(contract["driver_pass_status"], "PASS_ROUND14_A_DRIVER_RAW")
            self.assertEqual(contract["process_pass_status"], "PASS_ROUND14_A_MODEL_PROCESS")
            self.assertIn(str(version), contract["driver_command"])
            self.assertIn(backend, contract["driver_command"])
            self.assertIn(block, contract["driver_command"])
        with self.assertRaisesRegex(ValueError, "version"):
            api.contract(234)

    def test_patch_exact_rejects_drift_and_multiple_matches(self):
        api = load_module()
        self.assertEqual(api.patch_exact("a OLD b", "OLD", "NEW"), "a NEW b")
        with self.assertRaisesRegex(RuntimeError, "exactly once"):
            api.patch_exact("no match", "OLD", "NEW")
        with self.assertRaisesRegex(RuntimeError, "exactly once"):
            api.patch_exact("OLD OLD", "OLD", "NEW")

    def test_kt_control_plane_has_round14_only_identity(self):
        api = load_module()
        parent = ROOT / "artifacts/round14/parents/round13-final-v228"
        values = api.render_kt_control_plane(parent, api.contract(235))
        launcher = values["source/tools/launch_hybrid.py"]
        monitor = values["source/tools/monitor_owned_server.py"]
        controller = values["round14_a_controller.py"]
        supervisor = values["round14_a_supervisor.py"]
        self.assertIn("a.version != 235", launcher)
        self.assertIn("kt-round14-a-k-a-v235", launcher)
        self.assertIn("20260927-round14-a-k-a-v235", launcher)
        self.assertIn("ROUND14_A_KT_NATIVE_TP8_SYSTEM_COMPARISON", launcher)
        self.assertIn("kt-round14-a-k-[ab]-v(?:235|238)", monitor)
        self.assertIn("ROUND14_A_CONTRACT.json", controller)
        self.assertIn("round14-source-sha256.json", controller)
        self.assertIn("round14-a-supervisor.json", supervisor)
        self.assertIn("contract[\"generation_limit\"]", supervisor)
        self.assertIn("contract[\"process_pass_status\"]", supervisor)

    def test_native_control_plane_has_round14_only_identity(self):
        api = load_module()
        parent = ROOT / "artifacts/round12a/native_tp8/offline-snapshots-r2/v180"
        values = api.render_native_control_plane(parent, api.contract(239))
        launcher = values["native_tp8_launcher.py"]
        monitor = values["native_tp8_monitor.py"]
        admission = values["native_tp8_admission.py"]
        controller = values["round14_a_controller.py"]
        supervisor = values["round14_a_supervisor.py"]
        self.assertIn("version != 239", launcher)
        self.assertIn("kt-round14-a-n-{contract['block'].lower()}-v{version}", launcher)
        self.assertIn('("kt-round10", "kt-round11", "kt-round12", "kt-round13", "kt-round14")', launcher)
        self.assertIn("kt-round14-a-n-a-v239", monitor)
        self.assertIn('name.startswith("kt-round14")', admission)
        self.assertIn("ROUND14_A_CONTRACT.json", controller)
        self.assertIn("round14-source-sha256.json", controller)
        self.assertIn("contract[\"generation_limit\"]", supervisor)
        self.assertNotIn("raw.get(\"generation_attempts\") != 4", supervisor)
        self.assertIn("contract[\"process_pass_status\"]", supervisor)

    def test_build_writes_immutable_manifest_and_delta_receipt(self):
        api = load_module()
        parent = ROOT / "artifacts/round14/parents/round13-final-v228"
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "v235"
            report = api.build(parent, output, 235, ROOT / "artifacts/round14/round14_a_driver.py")
            self.assertEqual(report["status"], "PASS_ROUND14_A_SNAPSHOT_BUILD")
            self.assertEqual(report["version"], 235)
            manifest = json.loads((output / "round14-source-sha256.json").read_text())
            self.assertEqual(manifest, api.tree(output))
            self.assertNotIn("execution-budget.json", manifest)
            self.assertNotIn("round14-source-sha256.json", manifest)
            self.assertEqual(
                json.loads((output / "ROUND14_A_CONTRACT.json").read_text())["version"],
                235,
            )
            delta = json.loads((output / "ROUND14_A_SOURCE_DELTA.json").read_text())
            self.assertEqual(delta["parent_runtime_sha256"],
                             "eb51e7b0073d397673887b0e0fe6e41d745b8ee72a385faf7f1c28cf78b23fd2")
            self.assertEqual(delta["model_runtime_change"], "NONE")

    def test_native_repair_copies_every_frozen_driver_dependency(self):
        api = load_module()
        parent = ROOT / "artifacts/round12a/native_tp8/offline-snapshots-r2/v180"
        fixture11 = ROOT / "artifacts/round14/parents/round13-final-v228/round11-t1-fixture.json"
        protocol10 = ROOT / "artifacts/round14/parents/round13-final-v228/round10_t1_protocol.py"
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "v251"
            api.build(
                parent, output, 251,
                ROOT / "artifacts/round14/round14_a_driver.py",
                fixture11=fixture11,
                protocol10=protocol10,
            )
            contract = json.loads((output / "ROUND14_A_CONTRACT.json").read_text())
            for argument in contract["driver_command"]:
                if argument.startswith("{DRIVER_ROOT}/"):
                    relative = argument.removeprefix("{DRIVER_ROOT}/")
                    self.assertTrue((output / relative).is_file(), relative)
            self.assertEqual(
                (output / "round11-t1-fixture.json").read_bytes(),
                fixture11.read_bytes(),
            )
            self.assertEqual(
                (output / "round10_t1_protocol.py").read_bytes(),
                protocol10.read_bytes(),
            )
            self.assertEqual(
                json.loads((output / "ROUND14_A_SOURCE_DELTA.json").read_text())[
                    "round10_protocol_sha256"
                ],
                "3b5d96864922ad4bd51d298ebefd039fe06e2de9a1bef8de4f800c8dcf2e0129",
            )


if __name__ == "__main__":
    unittest.main()
