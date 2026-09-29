from __future__ import annotations

import importlib
import sys
import unittest


class Q32HybridLiveProbeTests(unittest.TestCase):
    def test_import_has_no_device_runtime_side_effect(self):
        before = set(sys.modules)
        module = importlib.import_module(
            "round16_integration.b_q32_hybrid_live_probe"
        )
        self.assertEqual(
            module.Q32_V3_PROBE_API,
            "round16-integration:q32-hybrid-live-probe:v1",
        )
        newly_loaded = set(sys.modules) - before
        self.assertNotIn("torch", newly_loaded)
        self.assertNotIn("torch_npu", newly_loaded)
        self.assertNotIn("vllm", newly_loaded)
        self.assertNotIn("vllm_ascend", newly_loaded)

    def test_device_gate_is_explicit(self):
        module = importlib.import_module(
            "round16_integration.b_q32_hybrid_live_probe"
        )
        with self.assertRaises(RuntimeError):
            module._require_device_enable({})
        module._require_device_enable({"ROUND16_B_ENABLE_Q32_V3": "1"})


if __name__ == "__main__":
    unittest.main()
