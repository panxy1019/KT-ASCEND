import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
MODULE = ROOT / "artifacts/round14/build_round14_c_profile_snapshot.py"


def load_module():
    spec = importlib.util.spec_from_file_location("build_round14_c_profile_snapshot", MODULE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Round14CProfileSnapshotBuilderTest(unittest.TestCase):
    def test_contract_freezes_two_fresh_four_request_blocks(self):
        api = load_module()
        for version, block in ((249, "A"), (250, "B"), (274, "A"), (275, "B")):
            value = api.contract(version)
            self.assertEqual(value["block"], block)
            self.assertEqual(value["generation_limit"], 4)
            self.assertEqual(value["profile_threshold"], 513)
            self.assertEqual(value["default_threshold"], 129)
            self.assertEqual(value["driver_pass_status"], "PASS_ROUND14_C_PROFILE_DRIVER_RAW")
            self.assertEqual(value["process_pass_status"], "PASS_ROUND14_C_PROFILE_PROCESS")

    def test_scheduler_patch_is_bounded_and_rejects_drift(self):
        api = load_module()
        parent = ROOT / "artifacts/round14/a-snapshots/v235"
        source = (parent / "overlay-python/sglang/srt/managers/scheduler.py").read_text()
        patched = api.patch_scheduler(source)
        self.assertIn("round13_before(self, batch)", patched)
        self.assertIn("round13_after(self)", patched)
        with self.assertRaisesRegex(RuntimeError, "scheduler"):
            api.patch_scheduler(patched)

    def test_build_installs_observer_driver_control_and_manifest(self):
        api = load_module()
        parent = ROOT / "artifacts/round14/a-snapshots/v235"
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "v249"
            report = api.build(
                parent, output, 249,
                ROOT / "artifacts/round14/round14_c_profile_driver.py",
                ROOT / "artifacts/round13/round13_attribution.py",
            )
            self.assertEqual(report["status"], "PASS_ROUND14_C_PROFILE_SNAPSHOT_BUILD")
            contract = json.loads((output / "ROUND14_C_PROFILE_CONTRACT.json").read_text())
            self.assertEqual(contract["version"], 249)
            self.assertEqual(contract["generation_limit"], 4)
            launcher = (output / "source/tools/launch_hybrid.py").read_text()
            self.assertIn("ROUND13_ATTRIBUTION_ENABLE=1", launcher)
            self.assertIn("round14-c-spans.jsonl", launcher)
            self.assertIn("kt-round14-c-profile-a-v249", launcher)
            scheduler = output / "overlay-python/sglang/srt/managers/scheduler.py"
            self.assertIn("round13_before(self, batch)", scheduler.read_text())
            manifest = json.loads((output / "round14-source-sha256.json").read_text())
            self.assertEqual(manifest, api.tree(output))
            self.assertEqual(
                json.loads((output / "framework-profile.json").read_text())["prefill"]["threshold_tokens"],
                513,
            )
            self.assertEqual(
                json.loads((output / "streaming-profile.json").read_text())["threshold_tokens"],
                513,
            )


if __name__ == "__main__":
    unittest.main()
