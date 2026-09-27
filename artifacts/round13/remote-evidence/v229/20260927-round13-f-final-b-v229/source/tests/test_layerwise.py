"""CPU-only contracts for bounded route-grouped layerwise transport."""
from types import SimpleNamespace as NS
import unittest


class RecordingBackend:
    def __init__(self, fail_allocation=None, fail_sync=None, fail_abort=False):
        self.fail_allocation = fail_allocation
        self.fail_sync = fail_sync
        self.fail_abort = fail_abort
        self.prepared = None
        self.executed = []
        self.synchronized = []
        self.released = []
        self.aborted = []
        self.broken = []
        self.state = "READY"

    def preallocate(self, reservation):
        self.prepared = reservation

    def execute_group(self, layer, group, hidden_states, topk_weights, identity):
        self.executed.append((layer, group, identity))
        if identity.allocation_epoch == self.fail_allocation:
            raise RuntimeError("cancelled transfer")
        rows = []
        for row, remapped in zip(group.rows, group.remapped_ids):
            # A deterministic stand-in for weighted expert execution. It uses
            # original weights and logical IDs, so remap or row-order bugs show.
            value = sum(
                topk_weights[row][column] * group.logical_experts[physical]
                for column, physical in enumerate(remapped)
            )
            rows.append(hidden_states[row] + value)
        return rows

    def synchronize(self, identity):
        if identity.allocation_epoch == self.fail_sync:
            raise RuntimeError("sync failure")
        self.synchronized.append(identity)

    def release(self, identity):
        self.released.append(identity)

    def assemble(self, rows, hidden_states):
        return rows

    def abort(self, request_id):
        if self.fail_abort:
            raise RuntimeError("abort failure")
        self.aborted.append(request_id)

    def fail_stop(self, error):
        self.broken.append(error)
        self.state = "BROKEN_RELOAD_REQUIRED"


class TensorLikeRows(list):
    """Minimal device-tensor surface used to verify host route normalization."""
    def __init__(self, rows):
        super().__init__(rows)
        self.converted = False

    def detach(self):
        return self

    def cpu(self):
        return self

    def tolist(self):
        self.converted = True
        return [list(row) for row in self]


class LayerwiseGroupingTests(unittest.TestCase):
    def api(self):
        from migration_bridge.round9.runtime import layerwise
        return layerwise

    def test_contiguous_groups_keep_complete_routes_and_cover_rows_once(self):
        groups = self.api().group_topk_routes(
            [[4, 1], [1, 2], [7, 8], [8, 9]], capacity=3,
            experts=10, topk=2,
        )
        self.assertEqual(
            [(g.rows, g.logical_experts, g.remapped_ids) for g in groups],
            [
                ((0, 1), (1, 2, 4), ((2, 0), (0, 1))),
                ((2, 3), (7, 8, 9), ((0, 1), (1, 2))),
            ],
        )

    def test_repeated_experts_across_tokens_are_loaded_once_per_group(self):
        group, = self.api().group_topk_routes(
            [[5, 1], [5, 2], [1, 2]], capacity=3, experts=8, topk=2,
        )
        self.assertEqual(group.logical_experts, (1, 2, 5))
        self.assertEqual(group.remapped_ids, ((2, 0), (2, 1), (0, 1)))

    def test_device_tensor_like_routes_are_normalized_without_changing_grouping(self):
        routes = TensorLikeRows([[3, 0], [0, 2]])
        groups = self.api().group_topk_routes(
            routes,
            capacity=3, experts=4, topk=2,
        )
        self.assertTrue(routes.converted)
        self.assertEqual(groups[0].logical_experts, (0, 2, 3))
        self.assertEqual(groups[0].remapped_ids, ((2, 0), (0, 1)))

    def test_duplicate_within_token_and_invalid_routes_are_rejected(self):
        api = self.api()
        for rows, options in (
            ([[1, 1]], dict(capacity=2, experts=4, topk=2)),
            ([[1]], dict(capacity=2, experts=4, topk=2)),
            ([[1, 4]], dict(capacity=2, experts=4, topk=2)),
            ([[1, 2]], dict(capacity=1, experts=4, topk=2)),
            ([], dict(capacity=2, experts=4, topk=2)),
        ):
            with self.subTest(rows=rows, options=options), self.assertRaises(ValueError):
                api.group_topk_routes(rows, **options)


class LayerwiseLifecycleTests(unittest.TestCase):
    def api(self):
        from migration_bridge.round9.runtime import layerwise
        return layerwise

    def transport(self, backend, layers=2):
        api = self.api()
        reservation = api.make_reservation(
            capacity=3, hidden=4, intermediate=2,
            max_group_rows=3,
            allocator_limit_bytes=10_000,
            observed_base_peak_bytes=1_000,
            observed_workspace_bytes=500,
        )
        return api.LayerwiseTransport(
            backend, reservation, experts=8, topk=2, layers=layers,
        )

    def test_execution_preserves_topk_weights_and_original_row_order(self):
        backend = RecordingBackend()
        transport = self.transport(backend, layers=1)
        transport.begin("request-a", placement_epoch=7)
        result = transport.execute_layer(
            "request-a", 7, 0,
            hidden_states=[100.0, 200.0, 300.0],
            topk_ids=[[4, 1], [1, 2], [7, 6]],
            topk_weights=[[0.25, 0.75], [0.5, 0.5], [0.1, 0.9]],
        )
        self.assertEqual(result, [101.75, 201.5, 306.1])
        transport.finish("request-a", 7)
        self.assertEqual([i.allocation_epoch for i in backend.synchronized], [1, 2])
        self.assertEqual(backend.synchronized, backend.released)

    def test_tensor_like_weights_are_validated_but_original_object_reaches_backend(self):
        backend = RecordingBackend()
        transport = self.transport(backend, layers=1)
        ids = TensorLikeRows([[4, 1]])
        weights = TensorLikeRows([[0.25, 0.75]])
        transport.begin("request-a", placement_epoch=7)
        self.assertEqual(
            transport.execute_layer("request-a", 7, 0, [100.0], ids, weights),
            [101.75],
        )
        self.assertTrue(ids.converted)
        self.assertTrue(weights.converted)

    def test_stale_request_placement_and_incomplete_layer_sequence_are_rejected(self):
        backend = RecordingBackend()
        transport = self.transport(backend)
        transport.begin("request-a", placement_epoch=3)
        with self.assertRaises(RuntimeError):
            transport.execute_layer("other", 3, 0, [0], [[0, 1]], [[.5, .5]])
        with self.assertRaises(RuntimeError):
            transport.execute_layer("request-a", 4, 0, [0], [[0, 1]], [[.5, .5]])
        with self.assertRaises(RuntimeError):
            transport.execute_layer("request-a", 3, 1, [0], [[0, 1]], [[.5, .5]])
        with self.assertRaises(RuntimeError):
            transport.finish("request-a", 3)

    def test_cancel_synchronizes_and_releases_failed_allocation_before_abort(self):
        backend = RecordingBackend(fail_allocation=1)
        transport = self.transport(backend, layers=1)
        transport.begin("request-a", placement_epoch=2)
        with self.assertRaisesRegex(RuntimeError, "cancelled transfer"):
            transport.execute_layer(
                "request-a", 2, 0, [0], [[0, 1]], [[.5, .5]],
            )
        self.assertEqual([i.allocation_epoch for i in backend.synchronized], [1])
        self.assertEqual(backend.synchronized, backend.released)
        self.assertEqual(backend.aborted, ["request-a"])
        with self.assertRaises(RuntimeError):
            transport.execute_layer(
                "request-a", 2, 0, [0], [[0, 1]], [[.5, .5]],
            )

    def test_sync_failure_never_releases_allocation_and_poison_prevents_reuse(self):
        backend = RecordingBackend(fail_allocation=1, fail_sync=1)
        transport = self.transport(backend, layers=1)
        transport.begin("request-a", placement_epoch=2)
        with self.assertRaisesRegex(RuntimeError, "cancelled transfer") as caught:
            transport.execute_layer(
                "request-a", 2, 0, [0], [[0, 1]], [[.5, .5]],
            )
        self.assertIsNotNone(caught.exception.__cause__)
        self.assertIn("sync failure", str(caught.exception.__cause__))
        self.assertEqual(backend.released, [])
        self.assertEqual(len(backend.broken), 1)
        self.assertEqual(backend.state, "BROKEN_RELOAD_REQUIRED")
        self.assertEqual(transport.state, "BROKEN_RELOAD_REQUIRED")
        with self.assertRaisesRegex(RuntimeError, "BROKEN_RELOAD_REQUIRED"):
            transport.begin("new", placement_epoch=3)

    def test_abort_failure_poisons_transport_but_retains_execution_exception(self):
        backend = RecordingBackend(fail_allocation=1, fail_abort=True)
        transport = self.transport(backend, layers=1)
        transport.begin("request-a", placement_epoch=2)
        with self.assertRaisesRegex(RuntimeError, "cancelled transfer") as caught:
            transport.execute_layer(
                "request-a", 2, 0, [0], [[0, 1]], [[.5, .5]],
            )
        self.assertIsNotNone(caught.exception.__cause__)
        self.assertIn("abort failure", str(caught.exception.__cause__))
        self.assertEqual(len(backend.broken), 1)
        self.assertEqual(backend.state, "BROKEN_RELOAD_REQUIRED")
        self.assertEqual(transport.state, "BROKEN_RELOAD_REQUIRED")
        with self.assertRaisesRegex(RuntimeError, "BROKEN_RELOAD_REQUIRED"):
            transport.begin("new", placement_epoch=3)

    def test_explicit_abort_failure_poisons_transport_and_backend(self):
        backend = RecordingBackend(fail_abort=True)
        transport = self.transport(backend, layers=1)
        transport.begin("request-a", placement_epoch=2)
        with self.assertRaisesRegex(RuntimeError, "abort failure"):
            transport.abort("request-a", 2)
        self.assertEqual(backend.state, "BROKEN_RELOAD_REQUIRED")
        self.assertEqual(transport.state, "BROKEN_RELOAD_REQUIRED")
        with self.assertRaisesRegex(RuntimeError, "BROKEN_RELOAD_REQUIRED"):
            transport.begin("new", placement_epoch=3)

    def test_explicit_abort_prevents_reuse_and_new_begin_gets_new_allocation_epoch(self):
        backend = RecordingBackend()
        transport = self.transport(backend, layers=1)
        transport.begin("old", placement_epoch=0)
        transport.abort("old", 0)
        with self.assertRaises(RuntimeError):
            transport.execute_layer("old", 0, 0, [0], [[0, 1]], [[.5, .5]])
        transport.begin("new", placement_epoch=1)
        transport.execute_layer("new", 1, 0, [0], [[0, 1]], [[.5, .5]])
        self.assertEqual(backend.executed[-1][2].allocation_epoch, 1)

    def test_preallocation_accounts_bank_projection_workspace_and_base_peak(self):
        api = self.api()
        reservation = api.make_reservation(
            capacity=32, hidden=2560, intermediate=640,
            max_group_rows=32,
            allocator_limit_bytes=43_008 * 1024**2,
            observed_base_peak_bytes=39_000 * 1024**2,
            observed_workspace_bytes=512 * 1024**2,
        )
        self.assertEqual(reservation.private_bank_bytes, 300 * 1024**2)
        self.assertEqual(reservation.max_projection_bytes, 200 * 1024**2)
        self.assertEqual(reservation.reserve_bytes, 1012 * 1024**2)
        self.assertLessEqual(reservation.required_peak_bytes, reservation.allocator_limit_bytes)
        with self.assertRaises(ValueError):
            api.make_reservation(
                capacity=32, hidden=2560, intermediate=640,
                max_group_rows=32,
                allocator_limit_bytes=39_500 * 1024**2,
                observed_base_peak_bytes=39_000 * 1024**2,
                observed_workspace_bytes=512 * 1024**2,
            )
        for changed in (
            {'observed_workspace_bytes': 0},
            {'observed_base_peak_bytes': 0},
            {'allocator_limit_bytes': 43_009 * 1024**2},
            {'max_group_rows': 33},
        ):
            args = dict(
                capacity=32, hidden=2560, intermediate=640,
                max_group_rows=32,
                allocator_limit_bytes=43_008 * 1024**2,
                observed_base_peak_bytes=39_000 * 1024**2,
                observed_workspace_bytes=512 * 1024**2,
            )
            args.update(changed)
            with self.subTest(changed=changed), self.assertRaises(ValueError):
                api.make_reservation(**args)

    def test_reservation_row_bound_rejects_oversized_execution(self):
        backend = RecordingBackend()
        transport = self.transport(backend, layers=1)
        transport.begin("request-a", placement_epoch=0)
        with self.assertRaisesRegex(ValueError, "max_group_rows"):
            transport.execute_layer(
                "request-a", 0, 0,
                hidden_states=[0, 0, 0, 0],
                topk_ids=[[0, 1], [0, 1], [0, 1], [0, 1]],
                topk_weights=[[.5, .5]] * 4,
            )
        self.assertEqual(backend.executed, [])


class LayerwisePrivateBackendTests(unittest.TestCase):
    def api(self):
        from migration_bridge.round9.runtime import layerwise
        return layerwise

    def test_private_shadow_copies_only_verified_scalars_and_new_runner_config(self):
        api = self.api()
        template = NS(
            top_k=10, moe_ep_size=1, moe_ep_rank=0, moe_tp_size=1,
            moe_tp_rank=0, hidden_size=2560,
            intermediate_size_per_partition=640,
            num_fused_shared_experts=0,
            mutable_cache={'must': 'not alias'},
            unregistered_tensor=object(),
            _forward_hooks={'must': 'not alias'},
        )
        runner = NS(is_gated=True, activation='silu')
        class PrivateModule:
            def __init__(self): self._forward_hooks = {}
        shadow = api._build_private_shadow(template, runner, 32, PrivateModule)
        self.assertEqual(shadow.top_k, 10)
        self.assertEqual(shadow.num_experts, 32)
        self.assertEqual(shadow.num_local_experts, 32)
        self.assertIs(shadow.moe_runner_config, runner)
        for forbidden in ('mutable_cache', 'unregistered_tensor'):
            self.assertFalse(hasattr(shadow, forbidden))
        self.assertIsNot(shadow._forward_hooks, template._forward_hooks)
        self.assertEqual(shadow._forward_hooks, {})

    def test_copy_only_adapter_hides_maps_publication_and_controller_surface(self):
        api = self.api()
        class Inner:
            copied_bytes = 7
            def copy_experts(self, layer, moves): self.copied = (layer, moves)
            def synchronize(self): self.synced = True
            def publish(self, *args): raise AssertionError('must be unreachable')
        inner = Inner()
        adapter = api._CopyOnlyAdapter(inner)
        adapter.copy_experts(3, ((0, 9),))
        adapter.synchronize()
        self.assertEqual(inner.copied, (3, ((0, 9),)))
        self.assertTrue(inner.synced)
        self.assertEqual(adapter.copied_bytes, 7)
        for forbidden in ('publish', 'methods', 'layers', 'source'):
            self.assertFalse(hasattr(adapter, forbidden))


if __name__ == "__main__":
    unittest.main()
