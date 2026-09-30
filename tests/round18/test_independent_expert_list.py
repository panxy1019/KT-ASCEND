from __future__ import annotations

from types import SimpleNamespace
import unittest

from round18.independent_expert_list import (
    ComponentGraphLedger,
    IndependentExpertPayload,
    IndependentExpertSlot,
    IndependentListPool,
    validate_component_graph_receipt,
)


class _Tensor:
    def __init__(self, shape, pointer, *, base=None):
        self.shape = shape
        self.dtype = "torch.bfloat16"
        self.device = SimpleNamespace(type="npu")
        self._pointer = pointer
        self._base = base

    def data_ptr(self):
        return self._pointer


def _slot(index: int) -> IndependentExpertSlot:
    return IndependentExpertSlot(
        physical_slot=index,
        w13_destination=_Tensor((2560, 1280), 1000 + 2 * index),
        w2_destination=_Tensor((640, 2560), 1001 + 2 * index),
    )


def _payload(logical_id: int) -> IndependentExpertPayload:
    return IndependentExpertPayload(
        logical_expert_id=logical_id,
        w13_source=object(),
        w2_source=object(),
    )


class IndependentExpertListTests(unittest.TestCase):
    def test_rejects_parent_views_and_aliased_projection_storage(self):
        with self.assertRaisesRegex(ValueError, "independent storage"):
            IndependentExpertSlot(
                physical_slot=0,
                w13_destination=_Tensor((2560, 1280), 1, base=object()),
                w2_destination=_Tensor((640, 2560), 2),
            )
        with self.assertRaisesRegex(ValueError, "distinct storage"):
            IndependentExpertSlot(
                physical_slot=0,
                w13_destination=_Tensor((2560, 1280), 3),
                w2_destination=_Tensor((640, 2560), 3),
            )

    def test_two_projection_ready_precedes_same_address_map_publication(self):
        events = []
        maps = []
        pool = IndependentListPool(
            slots=(_slot(0), _slot(1)),
            initial_slot_to_logical=(0, 1),
            copy_projection=lambda **kw: events.append(
                ("copy", kw["projection"], kw["physical_slot"], kw["generation"])
            )
            or True,
            ready_barrier=lambda **kw: events.append(
                ("ready", kw["physical_slot"], kw["generation"])
            )
            or True,
            publish_map=lambda **kw: (
                maps.append((id(kw["mapping"]), tuple(kw["mapping"]), kw["generation"])),
                events.append(("publish", kw["physical_slot"], kw["generation"])),
                True,
            )[-1],
        )
        mapping_identity = id(pool.expert_map)

        receipt = pool.replace(physical_slot=0, payload=_payload(33), generation=1)

        self.assertEqual(
            events,
            [
                ("copy", "w13", 0, 1),
                ("copy", "w2", 0, 1),
                ("ready", 0, 1),
                ("publish", 0, 1),
            ],
        )
        self.assertEqual(maps[0][0], mapping_identity)
        self.assertEqual(pool.expert_map[0], -1)
        self.assertEqual(pool.expert_map[33], 0)
        self.assertEqual(receipt["published_generation"], 1)
        self.assertTrue(receipt["data_ptr_stable"])
        self.assertFalse(receipt["parent_constructed"])
        self.assertFalse(receipt["parent_rebuilt"])

    def test_failed_second_projection_never_publishes_or_advances_generation(self):
        publications = []

        def copy_projection(**kwargs):
            return kwargs["projection"] != "w2"

        pool = IndependentListPool(
            slots=(_slot(0),),
            initial_slot_to_logical=(0,),
            copy_projection=copy_projection,
            ready_barrier=lambda **_kwargs: True,
            publish_map=lambda **kwargs: publications.append(kwargs) or True,
        )
        with self.assertRaisesRegex(RuntimeError, "w2 copy"):
            pool.replace(physical_slot=0, payload=_payload(33), generation=1)

        self.assertEqual(publications, [])
        self.assertEqual(pool.published_generation, 0)
        self.assertEqual(pool.expert_map[0], 0)
        self.assertEqual(pool.expert_map[33], -1)

    def test_a_b_a_cycle_keeps_slot_addresses_and_monotonic_generation(self):
        pool = IndependentListPool(
            slots=(_slot(0),),
            initial_slot_to_logical=(0,),
            copy_projection=lambda **_kwargs: True,
            ready_barrier=lambda **_kwargs: True,
            publish_map=lambda **_kwargs: True,
        )
        before = pool.pointer_snapshot()
        first = pool.replace(physical_slot=0, payload=_payload(33), generation=1)
        second = pool.replace(physical_slot=0, payload=_payload(0), generation=2)

        self.assertEqual(before, pool.pointer_snapshot())
        self.assertEqual(first["replaced_logical_expert"], 0)
        self.assertEqual(second["replaced_logical_expert"], 33)
        self.assertEqual(pool.published_generation, 2)
        self.assertEqual(pool.expert_map[0], 0)
        self.assertEqual(pool.expert_map[33], -1)
        with self.assertRaisesRegex(ValueError, "next generation"):
            pool.replace(physical_slot=0, payload=_payload(34), generation=4)

    def test_component_graph_requires_replay_before_consumption_and_external_mutation(self):
        ledger = ComponentGraphLedger()
        ledger.begin_capture()
        ledger.end_capture(output_generation=0)
        with self.assertRaisesRegex(RuntimeError, "capture output"):
            ledger.consume_output(generation=0)
        with self.assertRaisesRegex(RuntimeError, "replay boundary"):
            ledger.begin_external_mutation()

        ledger.record_replay(output_generation=0)
        ledger.consume_output(generation=0)
        ledger.begin_external_mutation()
        ledger.end_external_mutation(new_generation=1)
        ledger.record_replay(output_generation=1)
        ledger.consume_output(generation=1)

        receipt = ledger.receipt()
        self.assertEqual(receipt["scope"], "COMPONENT_NPUGRAPH_ONLY")
        self.assertFalse(receipt["full_model_graph_claim"])
        self.assertTrue(receipt["capture_output_rejected"])
        validate_component_graph_receipt(receipt)


if __name__ == "__main__":
    unittest.main()
