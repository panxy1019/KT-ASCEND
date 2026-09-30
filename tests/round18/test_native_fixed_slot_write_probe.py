from __future__ import annotations

from types import SimpleNamespace
import unittest

from round18.fixed_slot_write import (
    FixedSlotBacking,
    FixedSlotWriter,
    ProjectionSource,
    SourceExpert,
)
from round18_integration.native_fixed_slot_write_probe import (
    MeasuredSlotHooks,
    build_ineligible_backing_receipt,
    inspect_postload_backing,
    run_fixed_slot_cycle,
)


class _Tensor:
    def __init__(self, shape, pointer, *, dtype="torch.bfloat16", device="npu:0"):
        self.shape = shape
        self.dtype = dtype
        self.device = SimpleNamespace(type=device.split(":", 1)[0])
        self._pointer = pointer

    def data_ptr(self):
        return self._pointer


def _source(logical_id: int) -> SourceExpert:
    return SourceExpert(
        logical_expert_id=logical_id,
        gate=ProjectionSource("gate", (640, 2560), "bfloat16", object()),
        up=ProjectionSource("up", (640, 2560), "bfloat16", object()),
        down=ProjectionSource("down", (2560, 640), "bfloat16", object()),
    )


class NativeFixedSlotWriteProbeTests(unittest.TestCase):
    def test_postload_nz29_is_inspected_and_rejected_without_mutation(self):
        owner = SimpleNamespace(
            w13_weight=_Tensor((32, 2560, 1280), 101),
            w2_weight=_Tensor((32, 640, 2560), 202),
        )
        inspection = inspect_postload_backing(owner, get_npu_format=lambda _value: 29)

        self.assertFalse(inspection["eligible"])
        self.assertEqual(inspection["w13"]["npu_format"], 29)
        self.assertEqual(inspection["w2"]["npu_format"], 29)
        self.assertEqual(inspection["data_ptrs"], {"w13": 101, "w2": 202})
        self.assertIn("preprocessed", inspection["failure_reason"])

        receipt = build_ineligible_backing_receipt(
            inspection=inspection,
            allocator={"before_bytes": 10, "after_bytes": 10, "peak_bytes": 10},
            format_wall_seconds=0.25,
        )
        self.assertEqual(
            receipt["status"], "FAIL_FIXED_SLOT_POSTLOAD_FORMAT_INELIGIBLE"
        )
        self.assertFalse(receipt["mutation_started"])
        self.assertFalse(receipt["full_parent_rebuild"])
        self.assertIsNone(receipt["performance_claim"])

    def test_cycle_injects_partial_failure_then_publishes_and_restores(self):
        events = []
        backing = FixedSlotBacking(
            physical_slot=3,
            physical_slot_count=32,
            dtype="bfloat16",
            backing_format="BF16_FIXED_Q32",
            gate_destination=object(),
            up_destination=object(),
            down_destination=object(),
        )

        def copy_projection(**kwargs):
            events.append(("copy", kwargs["projection"], kwargs["generation"]))
            return True

        def publish_barrier(**kwargs):
            events.append(("publish", kwargs["generation"]))
            return True

        hooks = MeasuredSlotHooks(
            copy_projection=copy_projection,
            publish_barrier=publish_barrier,
        )
        writer = FixedSlotWriter(
            backing=backing,
            copy_projection=hooks.copy_projection,
            publish_barrier=hooks.publish_barrier,
        )
        pointers = {"w13": 101, "w2": 202, "gate": 111, "up": 112, "down": 203}
        receipt = run_fixed_slot_cycle(
            writer=writer,
            hooks=hooks,
            replacement_source=_source(99),
            original_source=_source(3),
            verify_replacement=lambda: {"passed": True, "relative_l2": 0.001},
            verify_restored=lambda: {"passed": True, "relative_l2": 0.002},
            read_data_ptrs=lambda: dict(pointers),
            read_allocator=lambda: {
                "before_bytes": 100,
                "after_bytes": 100,
                "peak_bytes": 120,
            },
        )

        self.assertEqual(receipt["status"], "PASS_FIXED_SLOT_REPLACE_RESTORE")
        self.assertEqual(receipt["published_generations"], [1, 2])
        self.assertTrue(receipt["midwrite_failure_unpublished"])
        self.assertTrue(receipt["skipped_generation_rejected"])
        self.assertTrue(receipt["stale_generation_rejected"])
        self.assertTrue(receipt["data_ptr_stable"])
        self.assertFalse(receipt["full_parent_rebuild"])
        self.assertEqual(
            events,
            [
                ("copy", "gate", 1),
                ("copy", "gate", 1),
                ("copy", "up", 1),
                ("copy", "down", 1),
                ("publish", 1),
                ("copy", "gate", 2),
                ("copy", "up", 2),
                ("copy", "down", 2),
                ("publish", 2),
            ],
        )
        self.assertEqual(hooks.copy_attempts, 8)
        self.assertIn("copy_wall_seconds", receipt)
        self.assertIn("publish_wall_seconds", receipt)

    def test_cycle_rejects_pointer_change_even_if_numeric_checks_pass(self):
        backing = FixedSlotBacking(
            physical_slot=0,
            physical_slot_count=32,
            dtype="bfloat16",
            backing_format="BF16_FIXED_Q32",
            gate_destination=object(),
            up_destination=object(),
            down_destination=object(),
        )
        hooks = MeasuredSlotHooks(
            copy_projection=lambda **_kwargs: True,
            publish_barrier=lambda **_kwargs: True,
        )
        writer = FixedSlotWriter(
            backing=backing,
            copy_projection=hooks.copy_projection,
            publish_barrier=hooks.publish_barrier,
        )
        pointer_rows = iter(({"w13": 1, "w2": 2}, {"w13": 9, "w2": 2}))
        with self.assertRaisesRegex(RuntimeError, "data_ptr"):
            run_fixed_slot_cycle(
                writer=writer,
                hooks=hooks,
                replacement_source=_source(33),
                original_source=_source(0),
                verify_replacement=lambda: {"passed": True},
                verify_restored=lambda: {"passed": True},
                read_data_ptrs=lambda: next(pointer_rows),
                read_allocator=lambda: {
                    "before_bytes": 1,
                    "after_bytes": 1,
                    "peak_bytes": 1,
                },
            )

    def test_replacement_verification_failure_still_restores_original_slot(self):
        events = []
        backing = FixedSlotBacking(
            physical_slot=0,
            physical_slot_count=32,
            dtype="bfloat16",
            backing_format="BF16_FIXED_Q32",
            gate_destination=object(),
            up_destination=object(),
            down_destination=object(),
        )
        hooks = MeasuredSlotHooks(
            copy_projection=lambda **kwargs: events.append(
                (kwargs["projection"], kwargs["generation"])
            )
            or True,
            publish_barrier=lambda **kwargs: events.append(
                ("publish", kwargs["generation"])
            )
            or True,
        )
        writer = FixedSlotWriter(
            backing=backing,
            copy_projection=hooks.copy_projection,
            publish_barrier=hooks.publish_barrier,
        )
        with self.assertRaisesRegex(RuntimeError, "replacement verification"):
            run_fixed_slot_cycle(
                writer=writer,
                hooks=hooks,
                replacement_source=_source(33),
                original_source=_source(0),
                verify_replacement=lambda: {"passed": False},
                verify_restored=lambda: {"passed": True},
                read_data_ptrs=lambda: {"w13": 1, "w2": 2},
                read_allocator=lambda: {
                    "before_bytes": 1,
                    "after_bytes": 1,
                    "peak_bytes": 1,
                },
            )
        self.assertEqual(writer.published_generation, 2)
        self.assertEqual(
            events[-4:],
            [("gate", 2), ("up", 2), ("down", 2), ("publish", 2)],
        )


if __name__ == "__main__":
    unittest.main()
