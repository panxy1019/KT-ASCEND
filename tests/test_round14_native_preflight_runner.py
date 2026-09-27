import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
MODULE = ROOT / "artifacts/round14/run_native_same_image_preflight.py"


def load_module():
    spec = importlib.util.spec_from_file_location("round14_preflight_runner", MODULE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Round14NativePreflightRunnerTest(unittest.TestCase):
    def test_command_is_networkless_deviceless_and_mounts_model_read_only(self):
        api = load_module()
        command = api.docker_command(
            Path("/snapshot"), Path("/evidence"), image="frozen-image"
        )
        self.assertIn("--network", command)
        self.assertIn("none", command)
        self.assertNotIn("--device", command)
        self.assertIn("/snapshot:/snapshot:ro", command)
        self.assertIn("/evidence:/preflight", command)
        self.assertIn("/home/admin/models/Qwen3.8-Flash-Next:/models:ro", command)
        self.assertNotIn("ASCEND_RT_VISIBLE_DEVICES=0", " ".join(command))

    def test_result_gate_requires_old_qsa_tp8_no_devices_and_restoration(self):
        api = load_module()
        valid = {
            "status": "PASS_ROUND12A_NATIVE_TP8_SAME_IMAGE_PREFLIGHT",
            "physical_devices_exposed": [],
            "tp_size": 8,
            "qsa_sha256": api.OLD_QSA_SHA256,
            "no_device_import_probe": {
                "restored": True,
                "mem_get_info_restored": True,
                "get_device_name_restored": True,
                "memory_capacity_for_serverargs_mib": 65536,
            },
        }
        self.assertTrue(api.validate_result(valid))
        for key, replacement in (
            ("physical_devices_exposed", [0]),
            ("tp_size", 1),
            ("qsa_sha256", "wrong"),
        ):
            broken = json.loads(json.dumps(valid))
            broken[key] = replacement
            with self.assertRaisesRegex(RuntimeError, "identity"):
                api.validate_result(broken)

    def test_repair_versions_are_the_only_current_native_formal_arms(self):
        api = load_module()
        self.assertEqual(api.FORMAL_NATIVE_VERSIONS, (251, 252))


if __name__ == "__main__":
    unittest.main()
