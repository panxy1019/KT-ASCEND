import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
MODULE = ROOT / "artifacts/round14/build_round14_c_gdn_component.py"


def load_module():
    spec = importlib.util.spec_from_file_location(
        "build_round14_c_gdn_component", MODULE
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Round14CGDNComponentBuilderTest(unittest.TestCase):
    def test_contract_freezes_two_component_blocks(self):
        api = load_module()
        for version, block in (
            (276, "A"), (277, "B"), (278, "A"), (279, "B"),
            (280, "A"), (281, "B"), (282, "A"), (283, "B"),
            (284, "A"), (285, "B"), (286, "A"), (287, "B")
        ):
            value = api.contract(version)
            self.assertEqual(value["version"], version)
            self.assertEqual(value["block"], block)
            self.assertEqual(value["candidate_default"], "OFF")
            self.assertEqual(value["full_parent_improvement_gate"], 0.15)
            self.assertEqual(value["automatic_retries"], 0)

    def test_qwen_patch_is_default_off_and_rejects_drift(self):
        api = load_module()
        parent = ROOT / "artifacts/round14/c-profile-snapshots/v274"
        source = (
            parent
            / "overlay-python/sglang/srt/models/qwen4_exp.py"
        ).read_text()
        patched = api.patch_qwen(source)
        self.assertIn(
            'os.environ.get("ROUND14_GDN_BACKEND_ENABLE", "0") == "1"',
            patched,
        )
        self.assertIn("core = self.attn(", patched)
        self.assertIn("normalized = sigmoid_gated_rms_norm(", patched)
        self.assertNotIn("return Qwen3_5GatedDeltaNet.forward(", patched)
        with self.assertRaisesRegex(RuntimeError, "GDN flag"):
            api.patch_qwen(patched)

    def test_build_freezes_manifest_and_monitor_identity(self):
        api = load_module()
        parent = ROOT / "artifacts/round14/c-profile-snapshots/v274"
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "v276"
            report = api.build(
                parent=parent,
                output=output,
                version=276,
                probe=ROOT / "artifacts/round14/round14_c_gdn_component_probe.py",
                controller=ROOT
                / "artifacts/round14/round14_c_gdn_component_controller.py",
            )
            self.assertEqual(
                report["status"], "PASS_ROUND14_C_GDN_COMPONENT_SNAPSHOT_BUILD"
            )
            contract = json.loads(
                (output / "ROUND14_C_GDN_COMPONENT_CONTRACT.json").read_text()
            )
            self.assertEqual(contract["container"], "kt-round14-c-gdn-component-a-v276")
            qwen = (
                output / "overlay-python/sglang/srt/models/qwen4_exp.py"
            ).read_text()
            self.assertEqual(qwen.count("ROUND14_GDN_BACKEND_ENABLE"), 1)
            self.assertEqual(qwen.count("sigmoid_gated_rms_norm("), 1)
            monitor = (output / "monitor_owned_server.py").read_text()
            self.assertIn(
                "kt-round14-c-gdn-component-[ab]-v(?:276|277|278|279|280|281|282|283|284|285|286|287)",
                monitor,
            )
            self.assertTrue((output / "numa_policy.py").is_file())
            manifest = json.loads((output / "source-sha256.json").read_text())
            self.assertEqual(manifest, api.tree(output))


if __name__ == "__main__":
    unittest.main()
