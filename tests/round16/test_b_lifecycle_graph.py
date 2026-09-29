from __future__ import annotations

import importlib
import inspect
import unittest

from round16.common.contracts import StateLease


def _modules():
    names = (
        "round16.b.lifecycle_bridge",
        "round16.b.graph_boundary",
        "round16.b.routing_bridge",
    )
    try:
        return tuple(importlib.import_module(name) for name in names)
    except ModuleNotFoundError as error:
        raise AssertionError("Round16-B lifecycle, graph, and routing modules must exist") from error


class LifecycleAndGraphTests(unittest.TestCase):
    @staticmethod
    def _acquire(
        manager,
        request_id,
        completion_id,
        *,
        request_epoch,
        slot_id=0,
        allocation_generation=None,
    ):
        arguments = {
            "request_epoch": request_epoch,
            "batch_generation": 7,
            "layer_id": 11,
            "token_range": (4, 9),
            "completion_id": completion_id,
            "writable_state": ("moe.output",),
            "slot_id": slot_id,
        }
        if allocation_generation is not None:
            arguments["allocation_generation"] = allocation_generation
        return manager.acquire(request_id, **arguments)

    @staticmethod
    def _cpu_plan(routing):
        return routing.plan_routes(
            [[40, -1, -1, -1, -1, -1, -1, -1, -1, -1]],
            [[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]],
            resident_logical_ids=(),
        )

    def test_cancel_requires_drain_and_slot_reuse_rejects_late_result(self) -> None:
        lifecycle, _, _ = _modules()
        manager = lifecycle.RequestLeaseManager(slot_count=1)
        lease1 = self._acquire(
            manager, "request-1", "completion-1", request_epoch=3
        )
        self.assertIsInstance(lease1.state_lease, StateLease)
        self.assertEqual(
            lease1.identity,
            (
                "request-1",
                3,
                1,
                7,
                11,
                (4, 9),
                "completion-1",
                0,
            ),
        )
        task1 = manager.begin_task(lease1, "cpu-moe")
        manager.cancel(lease1)

        with self.assertRaisesRegex(RuntimeError, "drain"):
            self._acquire(manager, "request-2", "completion-2", request_epoch=4)
        self.assertFalse(manager.complete(task1, "late-cancelled-output"))
        self.assertEqual(manager.drain(lease1), ())

        lease2 = self._acquire(
            manager, "request-2", "completion-2", request_epoch=4
        )
        self.assertEqual(lease2.generation, lease1.generation + 1)
        self.assertFalse(manager.complete(task1, "late-after-reuse"))
        task2 = manager.begin_task(lease2, "cpu-moe")
        self.assertTrue(manager.complete(task2, "fresh-output"))
        self.assertEqual(manager.drain(lease2), ("fresh-output",))

    def test_two_graph_replays_submit_distinct_cpu_tasks_with_lease_identity(self) -> None:
        lifecycle, graph_module, routing = _modules()
        backend_module = importlib.import_module("round16.b.kt_expert_backend")
        manager = lifecycle.RequestLeaseManager(slot_count=1)
        lease = self._acquire(
            manager, "request-graph", "completion-graph", request_epoch=8
        )
        plan = self._cpu_plan(routing)
        submitted_identities = []
        cpu_tasks = []

        def worker(task):
            cpu_tasks.append(task)
            return tuple((float(row[0]) * 10.0,) for row in task.hidden_rows)

        backend = backend_module.KTExpertBackend(worker, chunk_size=2)

        def submit_cpu(route_plan, hidden_states, identity_key):
            submitted_identities.append(identity_key)
            return backend.execute(
                route_plan,
                hidden_states,
                identity_key=identity_key,
            )

        boundary = graph_module.HybridGraphBoundary(manager)
        self.assertIn("input_fingerprint", inspect.signature(boundary.replay).parameters)
        first = boundary.replay(
            lease,
            plan,
            ((3.0,),),
            input_fingerprint="sha256:input-a",
            replay_npu=lambda route_plan: "npu-output",
            submit_cpu=submit_cpu,
        )
        second = boundary.replay(
            lease,
            plan,
            ((4.0,),),
            input_fingerprint="sha256:input-b",
            replay_npu=lambda route_plan: "npu-output",
            submit_cpu=submit_cpu,
        )

        self.assertEqual((first.replay_epoch, second.replay_epoch), (1, 2))
        self.assertNotEqual(first.cpu_task.identity, second.cpu_task.identity)
        self.assertNotEqual(first.cpu_identity, second.cpu_identity)
        self.assertEqual(
            first.cpu_identity[-4:],
            ("replay_epoch", 1, "input_fingerprint", "sha256:input-a"),
        )
        self.assertEqual(
            second.cpu_identity[-4:],
            ("replay_epoch", 2, "input_fingerprint", "sha256:input-b"),
        )
        self.assertEqual(first.cpu_task.identity[:len(lease.identity)], lease.identity)
        self.assertEqual(second.cpu_task.identity[:len(lease.identity)], lease.identity)
        self.assertEqual(submitted_identities, [first.cpu_identity, second.cpu_identity])
        self.assertEqual(cpu_tasks[0].identity_key[:len(first.cpu_identity)], first.cpu_identity)
        self.assertEqual(cpu_tasks[1].identity_key[:len(second.cpu_identity)], second.cpu_identity)
        route_key = plan.cpu_routes[0].key
        self.assertEqual(first.cpu_handle[route_key], (30.0,))
        self.assertEqual(second.cpu_handle[route_key], (40.0,))
        self.assertEqual(boundary.device_validation.reason_code, "DEVICE_REQUIRED_NOT_RUN")
        self.assertTrue(boundary.device_validation.requires_device)

    def test_cancelled_lease_is_rejected_before_any_replay_side_effect(self) -> None:
        lifecycle, graph_module, routing = _modules()
        manager = lifecycle.RequestLeaseManager(slot_count=1)
        lease = self._acquire(manager, "request-cancel", "completion-cancel", request_epoch=9)
        manager.cancel(lease)
        effects = []
        boundary = graph_module.HybridGraphBoundary(manager)
        self.assertIn("input_fingerprint", inspect.signature(boundary.replay).parameters)

        with self.assertRaisesRegex(RuntimeError, "cancelled"):
            boundary.replay(
                lease,
                self._cpu_plan(routing),
                ((1.0,),),
                input_fingerprint="sha256:cancelled",
                replay_npu=lambda plan: effects.append("npu"),
                submit_cpu=lambda plan, states, identity: effects.append("cpu"),
            )

        self.assertEqual(effects, [])
        self.assertEqual(manager.drain(lease), ())

    def test_empty_input_fingerprint_is_rejected_before_any_side_effect(self) -> None:
        lifecycle, graph_module, routing = _modules()
        manager = lifecycle.RequestLeaseManager(slot_count=1)
        lease = self._acquire(
            manager,
            "request-empty-fingerprint",
            "completion-empty-fingerprint",
            request_epoch=17,
        )
        effects = []

        with self.assertRaisesRegex(ValueError, "input_fingerprint"):
            graph_module.HybridGraphBoundary(manager).replay(
                lease,
                self._cpu_plan(routing),
                ((1.0,),),
                input_fingerprint="",
                replay_npu=lambda plan: effects.append("npu"),
                submit_cpu=lambda plan, states, identity: effects.append("cpu"),
            )

        self.assertEqual(effects, [])
        self.assertEqual(manager.drain(lease), ())

    def test_npu_replay_failure_retires_reserved_cpu_task(self) -> None:
        lifecycle, graph_module, routing = _modules()
        manager = lifecycle.RequestLeaseManager(slot_count=1)
        lease = self._acquire(manager, "request-npu-fail", "completion-npu-fail", request_epoch=10)
        cpu_effects = []
        boundary = graph_module.HybridGraphBoundary(manager)
        self.assertIn("input_fingerprint", inspect.signature(boundary.replay).parameters)

        def fail_npu(_plan):
            raise RuntimeError("injected npu replay failure")

        with self.assertRaisesRegex(RuntimeError, "injected npu"):
            boundary.replay(
                lease,
                self._cpu_plan(routing),
                ((1.0,),),
                input_fingerprint="sha256:npu-fail",
                replay_npu=fail_npu,
                submit_cpu=lambda plan, states, identity: cpu_effects.append(identity),
            )

        self.assertEqual(cpu_effects, [])
        self.assertEqual(manager.drain(lease), ())

    def test_cpu_submit_failure_retires_pending_task_for_slot_reuse(self) -> None:
        lifecycle, graph_module, routing = _modules()
        manager = lifecycle.RequestLeaseManager(slot_count=1)
        lease = self._acquire(manager, "request-cpu-fail", "completion-cpu-fail", request_epoch=11)
        npu_effects = []
        boundary = graph_module.HybridGraphBoundary(manager)
        self.assertIn("input_fingerprint", inspect.signature(boundary.replay).parameters)

        def fail_cpu(_plan, _states, _identity):
            raise RuntimeError("injected cpu submit failure")

        with self.assertRaisesRegex(RuntimeError, "injected cpu"):
            boundary.replay(
                lease,
                self._cpu_plan(routing),
                ((1.0,),),
                input_fingerprint="sha256:cpu-fail",
                replay_npu=lambda plan: npu_effects.append("npu") or "npu-output",
                submit_cpu=fail_cpu,
            )

        self.assertEqual(npu_effects, ["npu"])
        self.assertEqual(manager.drain(lease), ())
        reused = self._acquire(
            manager,
            "request-after-fail",
            "completion-after-fail",
            request_epoch=12,
        )
        self.assertGreater(reused.generation, lease.generation)

    def test_native_allocation_generation_must_strictly_advance(self) -> None:
        lifecycle, _, _ = _modules()
        manager = lifecycle.RequestLeaseManager(slot_count=1)
        self.assertIn(
            "allocation_generation",
            inspect.signature(manager.acquire).parameters,
        )
        lease5 = self._acquire(
            manager,
            "request-generation-5",
            "completion-generation-5",
            request_epoch=13,
            allocation_generation=5,
        )
        self.assertEqual(lease5.generation, 5)
        self.assertEqual(manager.drain(lease5), ())

        with self.assertRaisesRegex(ValueError, "strictly greater"):
            self._acquire(
                manager,
                "request-generation-repeat",
                "completion-generation-repeat",
                request_epoch=14,
                allocation_generation=5,
            )

        lease9 = self._acquire(
            manager,
            "request-generation-9",
            "completion-generation-9",
            request_epoch=15,
            allocation_generation=9,
        )
        self.assertEqual(lease9.generation, 9)

    def test_invalid_state_lease_does_not_consume_native_generation(self) -> None:
        lifecycle, _, _ = _modules()
        manager = lifecycle.RequestLeaseManager(slot_count=1)
        self.assertIn(
            "allocation_generation",
            inspect.signature(manager.acquire).parameters,
        )

        with self.assertRaisesRegex(ValueError, "non-negative"):
            manager.acquire(
                "request-invalid-epoch",
                request_epoch=-1,
                allocation_generation=6,
                batch_generation=7,
                layer_id=11,
                token_range=(4, 9),
                completion_id="completion-invalid-epoch",
                slot_id=0,
            )
        with self.assertRaisesRegex(ValueError, "token_range"):
            manager.acquire(
                "request-invalid-range",
                request_epoch=1,
                allocation_generation=6,
                batch_generation=7,
                layer_id=11,
                token_range=(9, 4),
                completion_id="completion-invalid-range",
                slot_id=0,
            )

        lease = self._acquire(
            manager,
            "request-valid-generation",
            "completion-valid-generation",
            request_epoch=16,
            allocation_generation=6,
        )
        self.assertEqual(lease.generation, 6)

    def test_identity_types_fail_before_active_or_generation_commit(self) -> None:
        lifecycle, _, _ = _modules()

        for slot_count in (True, 1.0, "1"):
            with self.subTest(slot_count=slot_count):
                with self.assertRaisesRegex(ValueError, "slot_count.*integer"):
                    lifecycle.RequestLeaseManager(slot_count=slot_count)

        cases = (
            (["request-list"], "completion", 0, "request_id.*string"),
            ("request", ["completion-list"], 0, "completion_id.*string"),
            ("request", "completion", True, "slot_id.*integer"),
        )
        for request_id, completion_id, slot_id, message in cases:
            with self.subTest(message=message):
                manager = lifecycle.RequestLeaseManager(slot_count=2)
                with self.assertRaisesRegex(ValueError, message):
                    manager.acquire(
                        request_id,
                        request_epoch=1,
                        allocation_generation=4,
                        batch_generation=2,
                        layer_id=3,
                        token_range=(0, 1),
                        completion_id=completion_id,
                        slot_id=slot_id,
                    )
                valid_slot = 1 if slot_id is True else 0
                lease = manager.acquire(
                    "request-valid",
                    request_epoch=1,
                    allocation_generation=4,
                    batch_generation=2,
                    layer_id=3,
                    token_range=(0, 1),
                    completion_id="completion-valid",
                    slot_id=valid_slot,
                )
                self.assertEqual(lease.generation, 4)
                self.assertEqual(manager.drain(lease), ())

    def test_task_purpose_requires_nonempty_string_without_consuming_sequence(self) -> None:
        lifecycle, _, _ = _modules()
        manager = lifecycle.RequestLeaseManager(slot_count=1)
        lease = self._acquire(
            manager,
            "request-purpose",
            "completion-purpose",
            request_epoch=18,
        )

        for purpose in (True, 1, ["cpu-moe"], ""):
            with self.subTest(purpose=purpose):
                with self.assertRaisesRegex(ValueError, "purpose.*non-empty string"):
                    manager.begin_task(lease, purpose)

        task = manager.begin_task(lease, "cpu-moe")
        self.assertEqual(task.sequence, 1)
        self.assertTrue(manager.abort_task(task))
        self.assertEqual(manager.drain(lease), ())


if __name__ == "__main__":
    unittest.main()
