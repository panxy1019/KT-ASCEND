from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

from round16.b.resident_packed_loader import ProjectionLoad
from round16.b.selective_loader import plan_q32
from round16_integration import b_q32_loader_probe as probe


class _Device:
    type = "npu"

    def __str__(self):
        return "npu:0"


class _Storage:
    def __init__(self, size):
        self._size = size

    def nbytes(self):
        return self._size


class _Tensor:
    def __init__(self, shape):
        self.shape = shape
        self.dtype = "torch.bfloat16"
        self.device = _Device()

    def numel(self):
        result = 1
        for value in self.shape:
            result *= value
        return result

    def element_size(self):
        return 2

    def untyped_storage(self):
        return _Storage(self.numel() * 2)


class Q32LoaderProbeTests(unittest.TestCase):
    def test_v1_receipt_is_a_hard_input_gate(self):
        valid = {
            "status": "REAL_IMAGE_Q32_CONSTRUCTOR_VERIFIED",
            "stage": "V1",
            "device_executed": True,
            "logical_experts": 512,
            "physical_npu_slots": 32,
            "payload_bytes": 314_572_800,
            "construction_method": "UnquantizedFusedMoEMethod",
            "loader_and_postload": "NOT_RUN_V2",
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "v1.json"
            path.write_text(json.dumps(valid))
            payload, digest = probe.load_v1_receipt(path)
            self.assertEqual(payload["stage"], "V1")
            self.assertEqual(len(digest), 64)
            valid["payload_bytes"] = 1
            path.write_text(json.dumps(valid))
            with self.assertRaisesRegex(ValueError, "payload_bytes"):
                probe.load_v1_receipt(path)

    def test_valid_v2_binding_promotes_only_loader_and_layout(self):
        plan = plan_q32(tuple(range(32)))
        owner = SimpleNamespace(
            round16_selective_plan=plan,
            round16_resident_logical_ids=plan.resident_logical_ids,
            w13_weight=_Tensor((32, 2560, 1280)),
            w2_weight=_Tensor((32, 640, 2560)),
        )
        coverage = tuple(
            ProjectionLoad(logical, slot, projection)
            for slot, logical in enumerate(plan.resident_logical_ids)
            for projection in ("gate", "up", "down")
        )
        reads = tuple(
            {
                "logical_expert_id": logical,
                "physical_slot": slot,
                "materialized_leading_experts": 1,
            }
            for projection in ("gate_up", "down")
            for slot, logical in enumerate(plan.resident_logical_ids)
        )
        cleanup = []
        binding = probe.Q32LoaderBinding(
            owner=owner,
            torch_module=None,
            coverage=coverage,
            construction_method="UnquantizedFusedMoEMethod",
            execution_method="AscendUnquantizedFusedMoEMethod",
            checkpoint_keys={"gate_up": "experts.gate_up_proj", "down": "experts.down_proj"},
            checkpoint_files={},
            checkpoint_shapes={
                "gate_up": (512, 1280, 2560),
                "down": (512, 2560, 640),
            },
            checkpoint_dtypes={"gate_up": "BF16", "down": "BF16"},
            source_row_reads=reads,
            selected_cpu_bytes=314_572_800,
            load_mode="SAFETENSORS_CPU_ROW_SLICE_NATIVE_PARAMETER_LOADER",
            postload_order="UPSTREAM_CREATE_LOAD_THEN_ASCEND_POSTLOAD",
            exact_value_checks={
                "gate_full_q32_exact": True,
                "up_full_q32_exact": True,
                "down_full_q32_exact": True,
            },
            w13_npu_format=29,
            w2_npu_format=29,
            worker_internal_format_assignment_issued=True,
            allocator_before_bytes=0,
            allocator_after_load_bytes=314_572_800,
            allocator_after_postload_bytes=314_572_800,
            allocator_peak_bytes=629_145_600,
            cleanup=lambda: cleanup.append(True),
        )
        snapshot = SimpleNamespace(
            resident_logical_ids=plan.resident_logical_ids,
        )

        receipt = probe.validate_and_release(
            binding,
            snapshot,
            layer_id=0,
            v1_receipt_sha256="a" * 64,
        )

        self.assertEqual(receipt["status"], "REAL_Q32_LOADER_AND_LAYOUT_VERIFIED")
        self.assertEqual(receipt["projection_coverage_count"], 96)
        self.assertFalse(receipt["hidden_full512_npu_payload"])
        self.assertEqual(receipt["hybrid_execution"], "NOT_RUN_V3")
        self.assertEqual(cleanup, [True])

    def test_full512_backing_storage_is_rejected_and_released(self):
        plan = plan_q32(tuple(range(32)))
        oversized = _Tensor((32, 2560, 1280))
        oversized.untyped_storage = lambda: _Storage(probe.FULL512_PAYLOAD_BYTES)
        owner = SimpleNamespace(
            round16_selective_plan=plan,
            round16_resident_logical_ids=plan.resident_logical_ids,
            w13_weight=oversized,
            w2_weight=_Tensor((32, 640, 2560)),
        )
        coverage = tuple(
            ProjectionLoad(logical, slot, projection)
            for slot, logical in enumerate(plan.resident_logical_ids)
            for projection in ("gate", "up", "down")
        )
        reads = tuple(
            {
                "logical_expert_id": logical,
                "materialized_leading_experts": 1,
            }
            for _ in range(2)
            for logical in plan.resident_logical_ids
        )
        cleanup = []
        binding = probe.Q32LoaderBinding(
            owner=owner,
            torch_module=None,
            coverage=coverage,
            construction_method="UnquantizedFusedMoEMethod",
            execution_method="AscendUnquantizedFusedMoEMethod",
            checkpoint_keys={},
            checkpoint_files={},
            checkpoint_shapes={"gate_up": (512, 1280, 2560), "down": (512, 2560, 640)},
            checkpoint_dtypes={"gate_up": "BF16", "down": "BF16"},
            source_row_reads=reads,
            selected_cpu_bytes=314_572_800,
            load_mode="SAFETENSORS_CPU_ROW_SLICE_NATIVE_PARAMETER_LOADER",
            postload_order="UPSTREAM_CREATE_LOAD_THEN_ASCEND_POSTLOAD",
            exact_value_checks={
                "gate_full_q32_exact": True,
                "up_full_q32_exact": True,
                "down_full_q32_exact": True,
            },
            w13_npu_format=29,
            w2_npu_format=29,
            worker_internal_format_assignment_issued=True,
            allocator_before_bytes=0,
            allocator_after_load_bytes=314_572_800,
            allocator_after_postload_bytes=314_572_800,
            allocator_peak_bytes=629_145_600,
            cleanup=lambda: cleanup.append(True),
        )
        with self.assertRaisesRegex(ValueError, "hidden full512"):
            probe.validate_and_release(
                binding,
                SimpleNamespace(resident_logical_ids=plan.resident_logical_ids),
                layer_id=0,
                v1_receipt_sha256="a" * 64,
            )
        self.assertEqual(cleanup, [True])


if __name__ == "__main__":
    unittest.main()
