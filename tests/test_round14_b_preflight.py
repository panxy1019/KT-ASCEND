import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
MODULE = ROOT / "artifacts/round14/round14_b_offline_preflight.py"
SNAPSHOT = ROOT / "artifacts/round14/b-snapshots/v257"


def load_module():
    spec = importlib.util.spec_from_file_location("round14_b_offline_preflight", MODULE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Round14BOfflinePreflightTest(unittest.TestCase):
    def test_preflight_proves_manifest_gate_profile_and_schedule_without_devices(self):
        api = load_module()
        report = api.preflight(SNAPSHOT, device_paths=[])
        self.assertEqual(report["status"], "PASS_ROUND14_B_OFFLINE_PREFLIGHT")
        self.assertEqual(report["physical_devices_exposed"], [])
        self.assertEqual(report["schedule_requests"], 64)
        self.assertEqual(report["hybrid_threshold_tokens"], 513)
        self.assertEqual(report["tokenizer_gate_calls"], 2)

    def test_preflight_rejects_any_exposed_device_or_source_drift(self):
        api = load_module()
        with self.assertRaisesRegex(RuntimeError, "device"):
            api.preflight(SNAPSHOT, device_paths=["/dev/davinci0"])
        with tempfile.TemporaryDirectory() as directory:
            clone = Path(directory) / "snapshot"
            import shutil
            shutil.copytree(SNAPSHOT, clone)
            path = clone / "ROUND14_H_BOUNDED_PROFILE.json"
            document = json.loads(path.read_text())
            document["max_input_tokens"] = 511
            path.write_text(json.dumps(document))
            with self.assertRaisesRegex(RuntimeError, "manifest"):
                api.preflight(clone, device_paths=[])

    def test_preflight_rejects_overlay_receipt_that_cannot_pass_launcher_gate(self):
        api = load_module()
        with tempfile.TemporaryDirectory() as directory:
            clone = Path(directory) / "snapshot"
            import shutil
            shutil.copytree(SNAPSHOT, clone)
            receipt_path = clone / "overlay-python/round9-candidate-overlay.json"
            receipt = json.loads(receipt_path.read_text())
            receipt["candidate_tree_sha256"] = "0" * 64
            receipt_path.write_text(json.dumps(receipt))
            (clone / "round14-source-sha256.json").write_text(
                json.dumps(api.tree(clone))
            )
            with self.assertRaisesRegex(RuntimeError, "overlay"):
                api.preflight(clone, device_paths=[])


if __name__ == "__main__":
    unittest.main()
