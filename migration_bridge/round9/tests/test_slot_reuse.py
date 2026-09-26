"""Host contracts for the opt-in same-layer private-shadow slot reuse."""
import json
from pathlib import Path
from types import SimpleNamespace as NS
import unittest

try:
    from migration_bridge.round9.runtime import slot_reuse
except ImportError:
    slot_reuse = None

from migration_bridge.round9.runtime.framework_config import BufferIdentity
from migration_bridge.round9.runtime.layerwise import (
    AscendBF16TransportBackend,
    LayerwiseTransport,
    RouteGroup,
    make_reservation,
)
from migration_bridge.round9.runtime.prefill_runtime import PrefillRuntime


class SlotReusePlannerTests(unittest.TestCase):
    def api(self):
        self.assertIsNotNone(slot_reuse, 'slot_reuse production module is required')
        return slot_reuse

    def test_hits_keep_physical_slots_and_all_consumer_ids_are_remapped(self):
        reuse = self.api().SameLayerSlotReuse(capacity=3)
        first = BufferIdentity('r', 4, 1)
        plan = reuse.prepare(7, (1, 2), ((1, 0),), first)
        self.assertEqual(plan.moves, ((0, 1), (1, 2)))
        self.assertEqual(plan.remapped_ids, ((1, 0),))
        self.assertEqual(reuse.snapshot()['committed_slots'], {})
        reuse.publish(first)

        second = BufferIdentity('r', 4, 2)
        plan = reuse.prepare(7, (2, 3), ((0, 1),), second)
        self.assertEqual(plan.moves, ((2, 3),))
        self.assertEqual(plan.remapped_ids, ((1, 2),))
        self.assertEqual(reuse.snapshot()['committed_slots'], {1: 0, 2: 1})
        reuse.publish(second)
        self.assertEqual(
            reuse.snapshot()['committed_slots'],
            {1: 0, 2: 1, 3: 2},
        )

    def test_layer_request_placement_or_epoch_discontinuity_clears_hits(self):
        changes = (
            ('other', 4, 2, 7),
            ('r', 5, 2, 7),
            ('r', 4, 4, 7),
            ('r', 4, 2, 8),
        )
        for request, placement, allocation, layer in changes:
            with self.subTest(change=(request, placement, allocation, layer)):
                reuse = self.api().SameLayerSlotReuse(capacity=3)
                first = BufferIdentity('r', 4, 1)
                reuse.prepare(7, (1, 2), ((0, 1),), first)
                reuse.publish(first)
                changed = BufferIdentity(request, placement, allocation)
                plan = reuse.prepare(layer, (2, 3), ((0, 1),), changed)
                self.assertEqual(plan.moves, ((0, 2), (1, 3)))
                self.assertEqual(plan.remapped_ids, ((0, 1),))

    def test_real_route_replay_matches_frozen_analysis_counts(self):
        root = Path(__file__).resolve().parents[3]
        tape = root / 'artifacts/round9-fixed-shadow32-update-20260925/v80/real-prefill.routes.jsonl'
        rows = [json.loads(line) for line in tape.read_text().splitlines()]
        reuse = self.api().SameLayerSlotReuse(capacity=32)
        for row in rows:
            if row.get('kind') != 'group':
                continue
            identity = BufferIdentity(
                row['request_id'], row['placement_epoch'], row['allocation_epoch']
            )
            reuse.prepare(
                row['logical_layer'], tuple(row['logical_experts']),
                tuple(tuple(route) for route in row['remapped_ids']), identity,
            )
            reuse.publish(identity)
        snapshot = reuse.snapshot()
        self.assertEqual(snapshot['requested_slots'], 20631)
        self.assertEqual(snapshot['cache_hits'], 5984)
        self.assertEqual(snapshot['changed_slots'], 14647)


class FakeArray:
    def __init__(self, value):
        self.value = value
        self.dtype = 'bf16'

    @property
    def shape(self):
        return (len(self.value),)

    def index_select(self, dimension, rows):
        return FakeArray([self.value[index] for index in rows.value])

    def __getitem__(self, index):
        return self.value[index]

    def __len__(self):
        return len(self.value)

    def __iter__(self):
        return iter(self.value)


class FakeTorch:
    int64 = 'int64'
    int32 = 'int32'

    def __init__(self):
        self.syncs = 0
        self.npu = NS(synchronize=self.synchronize)

    def synchronize(self, device):
        self.syncs += 1

    @staticmethod
    def tensor(value, **unused):
        return FakeArray(value)


class FakeAdapter:
    def __init__(self):
        self.calls = []
        self.fail = False

    def copy_experts(self, layer, moves):
        self.calls.append((layer, tuple(moves)))
        if self.fail:
            raise RuntimeError('partial copy')


class FakeMethod:
    def __init__(self):
        self.remapped = []

    def apply(self, shadow, dispatch):
        self.remapped.append(dispatch.topk.topk_ids.value)
        return NS(hidden_states=FakeArray([101] * len(dispatch.hidden.value)))


class Dispatch:
    def __init__(self, hidden, unused, topk):
        self.hidden = hidden
        self.topk = topk


class TopK:
    def __init__(self, weights, topk_ids, unused):
        self.weights = weights
        self.topk_ids = topk_ids


class SlotReuseBackendTests(unittest.TestCase):
    def backend(self, enabled=True):
        backend = AscendBF16TransportBackend(
            NS(), '/unused', capacity=32, layers=48, experts=512,
            enable_same_layer_slot_reuse=enabled,
        )
        backend.state = 'READY'
        backend.adapter = FakeAdapter()
        backend.shadow = object()
        backend.method = FakeMethod()
        backend.device = 'npu:0'
        backend._torch = FakeTorch()
        backend._StandardDispatchOutput = Dispatch
        backend._StandardTopKOutput = TopK
        return backend

    def test_enabled_backend_rejects_non_shadow32_or_non_all48_profile(self):
        for options in ({'capacity': 31, 'layers': 48},
                        {'capacity': 32, 'layers': 47}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                AscendBF16TransportBackend(
                    NS(), '/unused', experts=512,
                    enable_same_layer_slot_reuse=True, **options,
                )

    @staticmethod
    def execute(backend, identity, logical=(1, 2), remapped=((0, 1),)):
        group = RouteGroup((0,), logical, remapped)
        return backend.execute_group(
            7, group, FakeArray([10]), FakeArray([[.4, .6]]), identity
        )

    def test_default_off_copies_every_sorted_group_slot(self):
        backend = self.backend(enabled=False)
        identity = BufferIdentity('r', 0, 1)
        self.execute(backend, identity)
        self.assertEqual(backend.adapter.calls, [(7, ((0, 1), (1, 2)))])
        self.assertEqual(backend.slot_reuse_snapshot()['enabled'], False)

    def test_commit_occurs_only_after_existing_synchronize(self):
        backend = self.backend()
        first = BufferIdentity('r', 0, 1)
        self.execute(backend, first)
        before = backend.slot_reuse_snapshot()
        self.assertEqual(before['committed_slots'], {})
        self.assertEqual(before['pending_identity'], {
            'request_id': 'r', 'placement_epoch': 0, 'allocation_epoch': 1,
        })
        backend.synchronize(first)
        self.assertEqual(
            backend.slot_reuse_snapshot()['committed_slots'], {1: 0, 2: 1}
        )
        backend.release(first)

        second = BufferIdentity('r', 0, 2)
        self.execute(backend, second, (2, 3), ((0, 1),))
        self.assertEqual(backend.adapter.calls[-1], (7, ((2, 3),)))
        self.assertEqual(backend.method.remapped[-1], ((1, 2),))

    def test_abort_after_release_clears_committed_map_without_extra_sync(self):
        backend = self.backend()
        identity = BufferIdentity('r', 0, 1)
        self.execute(backend, identity)
        backend.synchronize(identity)
        backend.release(identity)
        self.assertNotEqual(backend.slot_reuse_snapshot()['committed_slots'], {})
        syncs = backend._torch.syncs
        backend.abort('r')
        self.assertEqual(backend.slot_reuse_snapshot()['committed_slots'], {})
        self.assertEqual(backend._torch.syncs, syncs)

    def test_stale_synchronize_invalidates_committed_map(self):
        backend = self.backend()
        identity = BufferIdentity('r', 0, 1)
        self.execute(backend, identity)
        backend.synchronize(identity)
        backend.release(identity)
        with self.assertRaisesRegex(RuntimeError, 'stale private shadow allocation'):
            backend.synchronize(BufferIdentity('r', 0, 2))
        self.assertEqual(backend.slot_reuse_snapshot()['committed_slots'], {})

    def test_partial_copy_invalidates_committed_and_pending_maps(self):
        backend = self.backend()
        first = BufferIdentity('r', 0, 1)
        self.execute(backend, first)
        backend.synchronize(first)
        backend.release(first)
        backend.adapter.fail = True
        second = BufferIdentity('r', 0, 2)
        with self.assertRaisesRegex(RuntimeError, 'partial copy'):
            self.execute(backend, second, (2, 3), ((0, 1),))
        snapshot = backend.slot_reuse_snapshot()
        self.assertEqual(snapshot['committed_slots'], {})
        self.assertIsNone(snapshot['pending_identity'])
        backend.synchronize(second)
        backend.release(second)
        self.assertEqual(backend.slot_reuse_snapshot()['committed_slots'], {})

    def test_prefill_failure_releases_then_aborts_and_poison_blocks_reuse(self):
        backend = self.backend()
        backend.preallocate = lambda reservation: None
        reservation = make_reservation(
            capacity=32, hidden=4, intermediate=2, max_group_rows=1,
            allocator_limit_bytes=100_000,
            observed_base_peak_bytes=1_000,
            observed_workspace_bytes=500,
        )
        transport = LayerwiseTransport(
            backend, reservation, experts=512, topk=2, layers=1
        )
        runtime = PrefillRuntime(
            transport, threshold_tokens=1, layers=1, experts=512, topk=2,
            max_tokens=1,
        )
        request = NS(
            rid='r', origin_input_ids=[1], last_extend_prefix_len=0,
            extend_input_len=1, finished_reason=None,
        )
        batch = NS(reqs=[request], forward_mode=NS(is_extend=lambda: True))
        self.assertTrue(runtime.before(batch, placement_epoch=0))
        aborts = []
        original_abort = backend.abort
        def abort(request_id):
            aborts.append((request_id, backend._current, backend._synchronized))
            return original_abort(request_id)
        backend.abort = abort
        backend.adapter.fail = True

        with self.assertRaisesRegex(RuntimeError, 'partial copy'):
            runtime.apply(
                0, FakeArray([10]), [[1, 2]], FakeArray([[.4, .6]])
            )

        self.assertEqual(aborts, [('r', None, None)])
        self.assertEqual(backend._torch.syncs, 1)
        self.assertIsNone(transport._active)
        self.assertIsNone(backend._current)
        self.assertIsNone(backend._synchronized)
        self.assertEqual(backend.slot_reuse_snapshot()['committed_slots'], {})
        self.assertEqual(runtime.state, 'BROKEN_RELOAD_REQUIRED')
        with self.assertRaisesRegex(RuntimeError, 'BROKEN_RELOAD_REQUIRED'):
            runtime.before(batch, placement_epoch=0)


if __name__ == '__main__':
    unittest.main()
