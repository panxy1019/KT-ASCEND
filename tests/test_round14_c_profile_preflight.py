import importlib.util
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
MODULE = ROOT / "artifacts/round14/round14_c_profile_offline_preflight.py"


def load_module():
    spec = importlib.util.spec_from_file_location("round14_c_profile_preflight", MODULE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Round14CProfilePreflightTest(unittest.TestCase):
    def test_v274_and_v275_pass_without_devices(self):
        api = load_module()
        for version, block in ((274, "A"), (275, "B")):
            report = api.preflight(
                ROOT / f"artifacts/round14/c-profile-snapshots/v{version}",
                physical_devices=[],
            )
            self.assertEqual(report["status"], "PASS_ROUND14_C_PROFILE_OFFLINE_PREFLIGHT")
            self.assertEqual(report["version"], version)
            self.assertEqual(report["block"], block)
            self.assertEqual(report["schedule_requests"], 4)
            self.assertEqual(report["physical_devices_exposed"], [])
            self.assertEqual(report["observer_device_query"], "DEFERRED_QUERY_ONLY_NO_SYNCHRONIZE")

    def test_preflight_rejects_any_exposed_device(self):
        api = load_module()
        with self.assertRaisesRegex(RuntimeError, "must not expose"):
            api.preflight(
                ROOT / "artifacts/round14/c-profile-snapshots/v274",
                physical_devices=["/dev/davinci0"],
            )


if __name__ == "__main__":
    unittest.main()
