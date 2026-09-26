import importlib.util
import unittest
from pathlib import Path


PATH = Path(__file__).resolve().parents[1] / "round10_t3_native_probe.py"
SPEC = importlib.util.spec_from_file_location("round10_t3_native_probe", PATH)
probe = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(probe)


class NativeProbeContractTests(unittest.TestCase):
    def test_frozen_identity_is_accepted(self):
        self.assertTrue(probe.validate_contract(
            capture_sha256=probe.CAPTURE_SHA256,
            bank_sha256=probe.BANK_SHA256,
            native_sha256=probe.NATIVE_SHA256,
            nodes=[2, 3, 4, 7],
        ))

    def test_topology_or_payload_drift_is_rejected(self):
        with self.assertRaises(ValueError):
            probe.validate_contract(
                capture_sha256=probe.CAPTURE_SHA256,
                bank_sha256=probe.BANK_SHA256,
                native_sha256=probe.NATIVE_SHA256,
                nodes=[2, 3, 6, 7],
            )


if __name__ == "__main__":
    unittest.main()
