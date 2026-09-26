"""CPU-only contracts for the isolated 320-slot workset component."""
from types import SimpleNamespace as NS
from dataclasses import replace
import unittest


class _Weight:
    def __init__(self, slots):
        self.shape = (slots, 1, 1)


class RecordingAdapter:
    copied_bytes = 0

    def __init__(self, fail_call=None):
        self.fail_call = fail_call
        self.calls = []

    def copy_experts(self, layer, moves):
        self.calls.append((layer, tuple(moves)))
        if len(self.calls) == self.fail_call:
            raise OSError("second segment failed")

    def synchronize(self):
        raise AssertionError("backend owns synchronization")

    def publish(self, *args):
        raise AssertionError("publication must remain unreachable")


class RecordingBackend:
    def __init__(self, adapter, *, fail_sync=False):
        self.adapter = adapter
        self.fail_sync = fail_sync
        self.capacity = 320
        self.state = "NEW"
        self.prepared = None
        self.consumer_calls = 0
        self.synchronized = []
        self.released = []
        self.aborted = []
        self.broken = []
        self._current = None

    def preallocate(self, reservation):
        self.prepared = reservation
        self.state = "READY"

    def execute_group(self, layer, group, hidden_states, topk_weights, identity):
        self._current = identity
        self.adapter.copy_experts(layer, tuple(enumerate(group.logical_experts)))
        self.consumer_calls += 1
        return [hidden_states[row] for row in group.rows]

    def synchronize(self, identity):
        self.synchronized.append(identity)
        if self.fail_sync:
            raise RuntimeError("cleanup synchronize failed")

    def release(self, identity):
        self.released.append(identity)
        self._current = None

    def assemble(self, rows, hidden_states):
        return rows

    def abort(self, request_id):
        self.aborted.append(request_id)

    def fail_stop(self, error):
        self.broken.append(error)
        self.state = "BROKEN_RELOAD_REQUIRED"


class ChunkWorksetComponentTests(unittest.TestCase):
    def api(self):
        from migration_bridge.round9.runtime import chunk_workset
        return chunk_workset

    def reservation(self):
        from migration_bridge.round9.runtime.layerwise import make_reservation
        return make_reservation(
            capacity=320, hidden=4, intermediate=2, max_group_rows=32,
            allocator_limit_bytes=100_000,
            observed_base_peak_bytes=1_000,
            observed_workspace_bytes=500,
        )

    def template(self, slots=32):
        return NS(w13_weight=_Weight(slots), w2_weight=_Weight(slots))

    def transport(self, inner):
        return self.api().make_component_transport(
            self.template(), "/unused", self.reservation(),
            layers=1, experts=512, topk=10,
            backend_factory=lambda *args, **kwargs: inner,
        )

    @staticmethod
    def routes_with_unique_count(count):
        rows = [list(range(start, start + 10)) for start in range(0, count - 10, 10)]
        rows.append(list(range(count - 10, count)))
        return rows

    def test_factory_keeps_resident_32_shadow_320_and_whole_parent_projection(self):
        adapter = RecordingAdapter()
        inner = RecordingBackend(adapter)
        transport = self.transport(inner)
        self.assertEqual(transport.backend.resident_capacity, 32)
        self.assertEqual(transport.backend.shadow_capacity, 320)
        self.assertEqual(inner.prepared.capacity, 320)
        # 320 * hidden * (2 * intermediate) * BF16 bytes: the complete w13
        # parent projection, not one 64-expert segment.
        self.assertEqual(inner.prepared.max_projection_bytes, 10_240)
        self.assertFalse(hasattr(inner.adapter, "publish"))

    def test_second_segment_failure_cleans_lease_then_poisons_backend_and_transport(self):
        adapter = RecordingAdapter(fail_call=2)
        inner = RecordingBackend(adapter)
        transport = self.transport(inner)
        routes = self.routes_with_unique_count(65)
        transport.begin("request-a", 7)

        with self.assertRaisesRegex(OSError, "second segment failed") as caught:
            transport.execute_layer(
                "request-a", 7, 0, [0] * len(routes), routes,
                [[0.1] * 10 for _ in routes],
            )

        self.assertEqual([len(moves) for _, moves in adapter.calls], [64, 1])
        self.assertEqual(inner.consumer_calls, 0)
        self.assertEqual(len(inner.synchronized), 1)
        self.assertEqual(inner.synchronized, inner.released)
        self.assertEqual(inner.aborted, [])
        self.assertEqual(inner.state, "BROKEN_RELOAD_REQUIRED")
        self.assertEqual(transport.backend.state, "BROKEN_RELOAD_REQUIRED")
        self.assertEqual(transport.state, "BROKEN_RELOAD_REQUIRED")
        self.assertIsInstance(caught.exception.__cause__, RuntimeError)
        with self.assertRaisesRegex(RuntimeError, "BROKEN_RELOAD_REQUIRED"):
            transport.begin("request-b", 8)

    def test_copy_failure_remains_primary_when_synchronous_cleanup_also_fails(self):
        adapter = RecordingAdapter(fail_call=2)
        inner = RecordingBackend(adapter, fail_sync=True)
        transport = self.transport(inner)
        routes = self.routes_with_unique_count(65)
        transport.begin("request-a", 7)

        with self.assertRaisesRegex(OSError, "second segment failed") as caught:
            transport.execute_layer(
                "request-a", 7, 0, [0] * len(routes), routes,
                [[0.1] * 10 for _ in routes],
            )

        self.assertIsNotNone(caught.exception.__cause__)
        self.assertIn("cleanup synchronize failed", str(caught.exception.__cause__))
        self.assertEqual(inner.released, [])
        self.assertEqual(transport.state, "BROKEN_RELOAD_REQUIRED")

    def test_successful_320_slot_group_is_segmented_before_one_consumer_call(self):
        adapter = RecordingAdapter()
        inner = RecordingBackend(adapter)
        transport = self.transport(inner)
        routes = [list(range(row * 10, (row + 1) * 10)) for row in range(32)]
        transport.begin("request-a", 0)

        result = transport.execute_layer(
            "request-a", 0, 0, list(range(32)), routes,
            [[0.1] * 10 for _ in routes],
        )

        self.assertEqual(result, list(range(32)))
        self.assertEqual([len(moves) for _, moves in adapter.calls], [64] * 5)
        self.assertEqual(inner.consumer_calls, 1)
        self.assertEqual(inner.synchronized, inner.released)

    def test_copy_segment_boundaries_never_exceed_64_experts(self):
        for count, expected in (
            (64, [64]), (65, [64, 1]), (128, [64, 64]),
            (129, [64, 64, 1]), (320, [64, 64, 64, 64, 64]),
        ):
            adapter = RecordingAdapter()
            segmented = self.api().SegmentedCopy(adapter)
            segmented.copy_experts(0, tuple((slot, slot) for slot in range(count)))
            with self.subTest(count=count):
                self.assertEqual(
                    [len(moves) for _, moves in adapter.calls], expected
                )

    def test_factory_rejects_non_32_resident_template_before_backend_creation(self):
        created = []
        with self.assertRaisesRegex(ValueError, "resident template"):
            self.api().make_component_transport(
                self.template(31), "/unused", self.reservation(),
                backend_factory=lambda *args, **kwargs: created.append(True),
            )
        self.assertEqual(created, [])

    def test_factory_rejects_forged_segment_sized_projection_reservation(self):
        reservation = self.reservation()
        forged = replace(
            reservation,
            max_projection_bytes=reservation.max_projection_bytes // 5,
            reserve_bytes=(reservation.reserve_bytes -
                           reservation.max_projection_bytes * 4 // 5),
        )
        created = []
        with self.assertRaisesRegex(ValueError, "whole-parent projection"):
            self.api().make_component_transport(
                self.template(), "/unused", forged,
                backend_factory=lambda *args, **kwargs: created.append(True),
            )
        self.assertEqual(created, [])


if __name__ == "__main__":
    unittest.main()
