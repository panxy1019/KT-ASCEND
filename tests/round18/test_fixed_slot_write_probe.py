from __future__ import annotations

import unittest

from round18_integration.fixed_slot_write_probe import (
    device_probe_parameter_schema,
    validate_device_probe_parameters,
)


class FixedSlotWriteProbeTests(unittest.TestCase):
    def test_schema_declares_real_ascend_requirements_without_performance_claim(self):
        schema = device_probe_parameter_schema()
        self.assertEqual(schema["api"], "round18:fixed-slot-write-probe-parameters:v1")
        self.assertEqual(schema["logical_experts"], 512)
        self.assertEqual(schema["physical_slots"], 32)
        self.assertEqual(schema["eligible_backing"], "BF16_FIXED_Q32")
        self.assertEqual(schema["ineligible_backings"], ["W8A8", "PREPROCESSED"])
        self.assertIsNone(schema["performance_claim"])
        self.assertIn("torch.npu.synchronize", schema["minimum_runtime_requirements"])

    def test_parameter_validation_rejects_bad_backing_or_missing_identity(self):
        valid = {
            "device": "npu:0",
            "logical_expert_id": 37,
            "physical_slot": 5,
            "generation": 1,
            "source_dtype": "bfloat16",
            "source_format": "BF16_CHECKPOINT_ROWS",
            "destination_dtype": "bfloat16",
            "destination_backing": "BF16_FIXED_Q32",
            "owner_api": "round16-b:source-bound-q32-routed-experts:offline-v1",
            "loader_api": "round16-b:resident-packed-loader:v1",
        }
        self.assertEqual(validate_device_probe_parameters(valid), valid)

        for bad in (
            {**valid, "destination_backing": "NZ29_PREPROCESSED"},
            {**valid, "destination_backing": "W8A8"},
            {key: value for key, value in valid.items() if key != "loader_api"},
        ):
            with self.subTest(bad=bad):
                with self.assertRaises((TypeError, ValueError)):
                    validate_device_probe_parameters(bad)


if __name__ == "__main__":
    unittest.main()
