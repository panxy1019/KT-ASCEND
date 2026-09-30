from __future__ import annotations

import unittest

from round18.fixed_slot_write import (
    FixedSlotBacking,
    FixedSlotWriter,
    ProjectionSource,
    SlotNotPublishedError,
    SourceExpert,
    UnsupportedBackingError,
)


def _source(logical_expert_id: int = 37) -> SourceExpert:
    return SourceExpert(
        logical_expert_id=logical_expert_id,
        gate=ProjectionSource("gate", (640, 2560), "bfloat16", object()),
        up=ProjectionSource("up", (640, 2560), "bfloat16", object()),
        down=ProjectionSource("down", (2560, 640), "bfloat16", object()),
    )


def _backing(*, backing_format: str = "BF16_FIXED_Q32") -> FixedSlotBacking:
    return FixedSlotBacking(
        physical_slot=5,
        physical_slot_count=32,
        dtype="torch.bfloat16",
        backing_format=backing_format,
        gate_destination=object(),
        up_destination=object(),
        down_destination=object(),
    )


class FixedSlotWriteTests(unittest.TestCase):
    def test_complete_slot_is_published_only_after_three_writes_and_barrier(self):
        events = []
        backing = _backing()

        def copy_projection(**kwargs):
            events.append(("copy", kwargs["projection"], kwargs["generation"]))
            self.assertIs(
                kwargs["destination"], backing.destination(kwargs["projection"])
            )
            return True

        def publish_barrier(**kwargs):
            events.append(("barrier", kwargs["physical_slot"], kwargs["generation"]))
            with self.assertRaises(SlotNotPublishedError):
                writer.acquire(generation=kwargs["generation"])
            return True

        writer = FixedSlotWriter(
            backing=backing,
            copy_projection=copy_projection,
            publish_barrier=publish_barrier,
        )
        receipt = writer.write(source=_source(), generation=1)

        self.assertEqual(
            events,
            [
                ("copy", "gate", 1),
                ("copy", "up", 1),
                ("copy", "down", 1),
                ("barrier", 5, 1),
            ],
        )
        self.assertEqual(receipt.logical_expert_id, 37)
        self.assertEqual(receipt.physical_slot, 5)
        self.assertEqual(receipt.generation, 1)
        self.assertEqual(receipt.projections_written, ("gate", "up", "down"))
        self.assertIs(writer.acquire(generation=1), backing)

    def test_partial_failure_never_publishes_and_same_generation_can_be_rewritten(self):
        events = []
        fail_up = True

        def copy_projection(**kwargs):
            nonlocal fail_up
            events.append((kwargs["projection"], kwargs["generation"]))
            if kwargs["projection"] == "up" and fail_up:
                fail_up = False
                raise RuntimeError("injected up failure")
            return True

        writer = FixedSlotWriter(
            backing=_backing(),
            copy_projection=copy_projection,
            publish_barrier=lambda **_kwargs: True,
        )

        with self.assertRaisesRegex(RuntimeError, "injected up failure"):
            writer.write(source=_source(), generation=1)
        with self.assertRaises(SlotNotPublishedError):
            writer.acquire(generation=1)

        receipt = writer.write(source=_source(), generation=1)
        self.assertEqual(receipt.generation, 1)
        self.assertEqual(
            events,
            [("gate", 1), ("up", 1), ("gate", 1), ("up", 1), ("down", 1)],
        )

    def test_stale_or_skipped_generation_is_rejected_before_any_write(self):
        copies = []
        writer = FixedSlotWriter(
            backing=_backing(),
            copy_projection=lambda **kwargs: copies.append(kwargs) or True,
            publish_barrier=lambda **_kwargs: True,
        )
        writer.write(source=_source(), generation=1)
        copies.clear()

        with self.assertRaisesRegex(ValueError, "expected generation 2"):
            writer.write(source=_source(38), generation=1)
        with self.assertRaisesRegex(ValueError, "expected generation 2"):
            writer.write(source=_source(38), generation=3)
        self.assertEqual(copies, [])
        self.assertIs(writer.acquire(generation=1), writer.backing)

        writer.write(source=_source(38), generation=2)
        with self.assertRaisesRegex(SlotNotPublishedError, "generation 1"):
            writer.acquire(generation=1)
        self.assertIs(writer.acquire(generation=2), writer.backing)

    def test_w8a8_and_preprocessed_backings_are_ineligible_before_write(self):
        for backing_format in ("W8A8", "NZ29_PREPROCESSED"):
            with self.subTest(backing_format=backing_format):
                copies = []
                writer = FixedSlotWriter(
                    backing=_backing(backing_format=backing_format),
                    copy_projection=lambda **kwargs: copies.append(kwargs) or True,
                    publish_barrier=lambda **_kwargs: True,
                )
                with self.assertRaises(UnsupportedBackingError):
                    writer.write(source=_source(), generation=1)
                self.assertEqual(copies, [])

    def test_source_requires_complete_dense_bf16_gate_up_down(self):
        writer = FixedSlotWriter(
            backing=_backing(),
            copy_projection=lambda **_kwargs: True,
            publish_barrier=lambda **_kwargs: True,
        )
        bad_dtype = SourceExpert(
            logical_expert_id=37,
            gate=ProjectionSource("gate", (640, 2560), "int8", object()),
            up=ProjectionSource("up", (640, 2560), "bfloat16", object()),
            down=ProjectionSource("down", (2560, 640), "bfloat16", object()),
        )
        with self.assertRaisesRegex(UnsupportedBackingError, "BF16"):
            writer.write(source=bad_dtype, generation=1)


if __name__ == "__main__":
    unittest.main()
