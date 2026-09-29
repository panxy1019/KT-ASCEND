from __future__ import annotations

import importlib
from dataclasses import replace
import unittest


def _module():
    try:
        return importlib.import_module("round16.b.source_bound_hybrid_runner")
    except ModuleNotFoundError as error:
        raise AssertionError("round16.b.source_bound_hybrid_runner must exist") from error


def _lease(lifecycle, *, completion_id="completion-1"):
    return lifecycle.acquire(
        "request-1",
        request_epoch=1,
        allocation_generation=1,
        batch_generation=1,
        layer_id=7,
        token_range=(0, 1),
        completion_id=completion_id,
        slot_id=0,
    )


class _RoutedOwner:
    def __init__(self, plan):
        self.round16_selective_plan = plan


class SourceBoundHybridRunnerTests(unittest.TestCase):
    def _fixture(self, *, ids=None, weights=None, cancel_during_cpu=False):
        module = _module()
        loader = importlib.import_module("round16.b.selective_loader")
        lifecycle_module = importlib.import_module("round16.b.lifecycle_bridge")
        backend_module = importlib.import_module("round16.b.kt_expert_backend")
        plan = loader.plan_q32(tuple(range(32)))
        lifecycle = lifecycle_module.RequestLeaseManager(slot_count=1)
        lease = _lease(lifecycle)
        calls = {"route": 0, "native": 0, "cpu": 0, "shared": 0}
        routed_ids = ids or ((0, 32, 1, 33, 2, 34, 3, 35, 4, 36),)
        routed_weights = weights or ((0.10,) * 10,)

        def route_once(**kwargs):
            calls["route"] += 1
            self.assertEqual(kwargs["top_k"], 10)
            self.assertEqual(kwargs["routed_scaling_factor"], 1.0)
            return module.FrozenTopK(
                routed_ids,
                routed_weights,
                valid_rows=(True,),
                routed_scaling_applied=False,
            )

        route_once.round16_api_identity = module.SOURCE_BOUND_ROUTE_ONCE_API

        def native_execute(route_plan, hidden_states, identity_key):
            calls["native"] += 1
            self.assertEqual(tuple(hidden_states), ((5.0, 6.0),))
            self.assertIn("input-1", identity_key)
            return {route.key: (1.0, 2.0) for route in route_plan.npu_routes}

        native_execute.round16_api_identity = (
            module.SOURCE_BOUND_NATIVE_PRESELECTED_API
        )

        def cpu_worker(task):
            calls["cpu"] += 1
            if cancel_during_cpu:
                lifecycle.cancel(lease)
            return tuple((3.0, 4.0) for _key in task.route_keys)

        backend = backend_module.KTExpertBackend(cpu_worker, chunk_size=32)

        def shared_execute(hidden_states, identity_key):
            calls["shared"] += 1
            self.assertIn("input-1", identity_key)
            return ((10.0, 20.0),)

        shared_execute.round16_api_identity = module.SOURCE_BOUND_SHARED_EXECUTE_API
        dependencies = module.SourceBoundRunnerDependencies(
            lifecycle=lifecycle,
            route_once=route_once,
            native_preselected_execute=native_execute,
            cpu_backend=backend,
            shared_execute=shared_execute,
        )
        binding = module.build_source_bound_hybrid_runner_binding(
            enabled=True,
            dependencies=dependencies,
        )
        runner = binding.runner_cls(
            layer_name="model.layers.7.mlp.experts",
            moe_config=object(),
            router=object(),
            routed_experts=_RoutedOwner(plan),
            shared_experts=object(),
            routed_scaling_factor=2.0,
            round16_selective_plan=plan,
            round16_runner_dependencies=dependencies,
        )
        return module, lifecycle, lease, binding, runner, calls

    def test_default_off_does_not_inspect_or_construct_dependencies(self) -> None:
        module = _module()
        receipt = module.build_source_bound_hybrid_runner_binding(enabled=False)
        self.assertEqual(receipt.status, "DISABLED_DEFAULT_OFF")
        self.assertIsNone(receipt.runner_cls)
        self.assertFalse(receipt.live_eligible)

    def test_legacy_decode_alone_requires_occurrence_mapping_callback(self) -> None:
        _module_value, _lifecycle, lease, _binding, runner, calls = self._fixture()
        runner._dependencies = replace(
            runner._dependencies,
            native_preselected_execute=None,
        )
        with self.assertRaisesRegex(RuntimeError, "mock-only"):
            runner.run_decode(
                lease,
                ((5.0, 6.0),),
                router_logits=((0.0,) * 512,),
                input_fingerprint="input-1",
            )
        self.assertEqual(calls["route"], 0)

    def test_routes_once_splits_q32_cpu_and_applies_weight_scale_shared_once(self) -> None:
        module, lifecycle, lease, binding, runner, calls = self._fixture()
        result = runner.run_decode(
            lease,
            ((5.0, 6.0),),
            router_logits=((0.0,) * 512,),
            input_fingerprint="input-1",
        )

        # Five resident outputs and five CPU outputs. Each route weight is 0.1,
        # the routed sum is scaled by 2 once, then shared is added once.
        self.assertEqual(result.output, ((14.0, 26.0),))
        self.assertEqual(calls, {"route": 1, "native": 1, "cpu": 5, "shared": 1})
        self.assertIsNotNone(result.cpu_task)
        self.assertIn("decode_epoch", result.work_id)
        self.assertFalse(binding.live_eligible)
        self.assertEqual(binding.status, "DEVICE_NATIVE_ABI_UNRESOLVED")
        self.assertEqual(runner.residual_owner, "CALLING_MODEL_LAYER")

        second = runner.run_decode(
            lease,
            ((5.0, 6.0),),
            router_logits=((0.0,) * 512,),
            input_fingerprint="input-1",
        )
        self.assertNotEqual(result.work_id, second.work_id)
        self.assertEqual(result.cpu_task.sequence + 1, second.cpu_task.sequence)
        drained = lifecycle.drain(lease)
        self.assertEqual(len(drained), 2)

    def test_all_resident_skips_cpu_task_but_still_routes_only_once(self) -> None:
        ids = (tuple(range(10)),)
        module, lifecycle, lease, _binding, runner, calls = self._fixture(ids=ids)
        result = runner.run_decode(
            lease,
            ((5.0, 6.0),),
            router_logits=((0.0,) * 512,),
            input_fingerprint="input-1",
        )
        self.assertIsNone(result.cpu_task)
        self.assertIsNone(result.work_id)
        self.assertEqual(calls, {"route": 1, "native": 1, "cpu": 0, "shared": 1})
        self.assertEqual(lifecycle.drain(lease), ())

    def test_cancel_before_route_and_late_cpu_completion_fail_closed(self) -> None:
        _module_value, lifecycle, lease, _binding, runner, calls = self._fixture()
        lifecycle.cancel(lease)
        with self.assertRaisesRegex(RuntimeError, "cancelled"):
            runner.run_decode(
                lease,
                ((5.0, 6.0),),
                router_logits=((0.0,) * 512,),
                input_fingerprint="input-1",
            )
        self.assertEqual(calls, {"route": 0, "native": 0, "cpu": 0, "shared": 0})
        self.assertEqual(lifecycle.drain(lease), ())

        _module_value, lifecycle, lease, _binding, runner, calls = self._fixture(
            cancel_during_cpu=True
        )
        with self.assertRaisesRegex(RuntimeError, "late or cancelled"):
            runner.run_decode(
                lease,
                ((5.0, 6.0),),
                router_logits=((0.0,) * 512,),
                input_fingerprint="input-1",
            )
        self.assertEqual(calls["route"], 1)
        self.assertEqual(calls["shared"], 0)
        self.assertEqual(lifecycle.drain(lease), ())

    def test_callback_failure_retires_cpu_task_and_invalid_route_scaling_fails(self) -> None:
        module, lifecycle, lease, _binding, runner, _calls = self._fixture()

        def fail_native(*args):
            raise RuntimeError("native source fixture failed")

        fail_native.round16_api_identity = module.SOURCE_BOUND_NATIVE_PRESELECTED_API
        runner._dependencies = module.SourceBoundRunnerDependencies(
            lifecycle=runner._dependencies.lifecycle,
            route_once=runner._dependencies.route_once,
            native_preselected_execute=fail_native,
            cpu_backend=runner._dependencies.cpu_backend,
            shared_execute=runner._dependencies.shared_execute,
        )
        with self.assertRaisesRegex(RuntimeError, "native source fixture"):
            runner.run_decode(
                lease,
                ((5.0, 6.0),),
                router_logits=((0.0,) * 512,),
                input_fingerprint="input-1",
            )
        self.assertEqual(lifecycle.drain(lease), ())

        with self.assertRaisesRegex(ValueError, "scaling"):
            module.FrozenTopK(
                ((0,) * 10,),
                ((0.1,) * 10,),
                valid_rows=(True,),
                routed_scaling_applied=True,
            )

    def test_unverified_callback_identity_is_rejected(self) -> None:
        module = _module()
        lifecycle_module = importlib.import_module("round16.b.lifecycle_bridge")
        backend_module = importlib.import_module("round16.b.kt_expert_backend")

        def callback(*args, **kwargs):
            return ()

        with self.assertRaisesRegex(ValueError, "route-once API identity"):
            module.SourceBoundRunnerDependencies(
                lifecycle=lifecycle_module.RequestLeaseManager(slot_count=1),
                route_once=callback,
                native_preselected_execute=callback,
                cpu_backend=backend_module.KTExpertBackend(callback),
                shared_execute=None,
            )

    def test_live_binding_names_exact_unresolved_native_seams(self) -> None:
        module, _lifecycle, _lease_value, binding, runner, _calls = self._fixture()
        self.assertFalse(binding.live_eligible)
        self.assertFalse(runner.runtime_runner_binding_verified)
        self.assertEqual(
            set(binding.unresolved_requirements),
            {
                "LIVE_QWEN_SELF_EXPERTS_IMAGE_PROOF",
                "EP1_ALLGATHER_Q32_PRESELECTED_DEVICE_ADAPTER",
                "Q32_NATIVE_OWNER_AND_RUNNER_DEVICE_PAIR",
                "S15_WEIGHTED_AGGREGATE_DEVICE_PARITY",
                "EXACT_SOURCE_TOKEN_ATOMIC_LIVE_REGISTRATION",
                "KT_NATIVE_STREAM_SINGLE_OUTSTANDING_DEVICE_DRAIN",
            },
        )

    def test_source_bound_kt_weighted_partial_is_not_weighted_or_scaled_twice(self):
        module = _module()
        loader = importlib.import_module("round16.b.selective_loader")
        lifecycle_module = importlib.import_module("round16.b.lifecycle_bridge")
        backend_module = importlib.import_module("round16.b.kt_expert_backend")
        lifecycle = lifecycle_module.RequestLeaseManager(slot_count=1)
        lease = _lease(lifecycle)
        plan = loader.plan_q32(tuple(range(32)))
        calls = {"route": 0, "native": 0, "submit": 0, "sync": 0}
        events = []
        ids = ((0, 32, 1, 33, 2, 34, 3, 35, 4, 36),)
        weights = ((0.1,) * 10,)

        def route_once(**_kwargs):
            calls["route"] += 1
            return module.FrozenTopK(ids, weights, valid_rows=(True,))

        route_once.round16_api_identity = module.SOURCE_BOUND_ROUTE_ONCE_API

        def native_execute(route_plan, _hidden_states, _identity_key):
            calls["native"] += 1
            self.assertEqual(events, ["submit"])
            events.append("native")
            return {route.key: (1.0, 2.0) for route in route_plan.npu_routes}

        native_execute.round16_api_identity = (
            module.SOURCE_BOUND_NATIVE_PRESELECTED_API
        )

        class Wrapper:
            def submit_forward(self, hidden_states, topk_ids, topk_weights, stream):
                calls["submit"] += 1
                events.append("submit")
                self.submitted = (hidden_states, topk_ids, topk_weights, stream)

            def sync_forward(self, ref_tensor, stream):
                calls["sync"] += 1
                events.append("sync")
                self.synced = (ref_tensor, stream)
                # This is already the sum of the five CPU routes multiplied by
                # their original 0.1 Top-K weights: 5 * (3, 4) * 0.1.
                return ((1.5, 2.0),)

        wrapper = Wrapper()

        def stream_handle(identity_key):
            self.assertIn("input-kt", identity_key)
            return "stream-handle"

        stream_handle.round16_api_identity = backend_module.KT_STREAM_HANDLE_API
        backend = backend_module.KTWeightedPartialBackend(
            wrapper,
            stream_handle_provider=stream_handle,
            api_identity=backend_module.KT_MOE_WRAPPER_FORWARD_API,
        )
        dependencies = module.SourceBoundRunnerDependencies(
            lifecycle=lifecycle,
            route_once=route_once,
            native_preselected_execute=native_execute,
            cpu_backend=backend,
            shared_execute=None,
        )
        runner = module.build_source_bound_hybrid_runner_binding(
            enabled=True, dependencies=dependencies
        ).runner_cls(
            layer_name="model.layers.7.mlp.experts",
            moe_config=object(),
            router=object(),
            routed_experts=_RoutedOwner(plan),
            routed_scaling_factor=2.0,
            round16_selective_plan=plan,
            round16_runner_dependencies=dependencies,
        )
        result = runner.run_decode(
            lease,
            ((5.0, 6.0),),
            router_logits=((0.0,) * 512,),
            input_fingerprint="input-kt",
        )
        # Native weighted partial=(0.5,1.0); KT partial=(1.5,2.0).
        # Their sum is scaled by two exactly once.
        self.assertEqual(result.output, ((4.0, 6.0),))
        self.assertEqual(calls, {"route": 1, "native": 1, "submit": 1, "sync": 1})
        self.assertEqual(wrapper.submitted[1], ids)
        self.assertEqual(wrapper.submitted[2], weights)
        self.assertEqual(wrapper.submitted[3], "stream-handle")
        self.assertEqual(events, ["submit", "native", "sync"])

        # If native execution fails after CPU submission, the pending KT work
        # is still synchronized/discarded and the lifecycle task is retired.
        self.assertEqual(len(lifecycle.drain(lease)), 1)
        second_lease = lifecycle.acquire(
            "request-2",
            request_epoch=2,
            allocation_generation=2,
            batch_generation=2,
            layer_id=7,
            token_range=(0, 1),
            completion_id="completion-2",
            slot_id=0,
        )

        def fail_native(*_args):
            calls["native"] += 1
            self.assertEqual(events, ["submit"])
            events.append("native")
            raise RuntimeError("native branch failed after CPU submit")

        fail_native.round16_api_identity = module.SOURCE_BOUND_NATIVE_PRESELECTED_API
        runner._dependencies = module.SourceBoundRunnerDependencies(
            lifecycle=lifecycle,
            route_once=route_once,
            native_preselected_execute=fail_native,
            cpu_backend=backend,
            shared_execute=None,
        )
        events.clear()
        with self.assertRaisesRegex(RuntimeError, "native branch failed"):
            runner.run_decode(
                second_lease,
                ((5.0, 6.0),),
                router_logits=((0.0,) * 512,),
                input_fingerprint="input-kt",
            )
        self.assertEqual(events, ["submit", "native", "sync"])
        self.assertEqual(lifecycle.drain(second_lease), ())

    def test_kt_native_without_work_id_allows_only_one_outstanding(self) -> None:
        backend_module = importlib.import_module("round16.b.kt_expert_backend")
        routing = importlib.import_module("round16.b.routing_bridge")
        plan = routing.plan_routes(
            ((32, -1, -1, -1, -1, -1, -1, -1, -1, -1),),
            ((1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),),
            resident_logical_ids=tuple(range(32)),
            logical_expert_count=512,
        )
        events = []

        class Wrapper:
            def submit_forward(self, *_args):
                events.append("submit")

            def sync_forward(self, *_args):
                events.append("sync")
                return ((2.0, 3.0),)

        def stream_handle(_identity_key):
            return "stream"

        stream_handle.round16_api_identity = backend_module.KT_STREAM_HANDLE_API
        backend = backend_module.KTWeightedPartialBackend(
            Wrapper(),
            stream_handle_provider=stream_handle,
            api_identity=backend_module.KT_MOE_WRAPPER_FORWARD_API,
        )
        pending = backend.submit(plan, ((1.0, 2.0),), identity_key=("work-1",))
        forged = backend_module.KTPendingWeightedPartial(
            pending.sequence + 1,
            ("work-2",),
            pending.token_count,
            pending.hidden_states,
            pending.stream_handle,
        )
        with self.assertRaisesRegex(RuntimeError, "stale"):
            backend.sync(forged)
        with self.assertRaisesRegex(RuntimeError, "one outstanding"):
            backend.submit(plan, ((1.0, 2.0),), identity_key=("work-2",))
        self.assertEqual(events, ["submit"])
        self.assertTrue(backend.drain_discard(pending))
        self.assertEqual(events, ["submit", "sync"])

        next_pending = backend.submit(
            plan, ((1.0, 2.0),), identity_key=("work-3",)
        )
        self.assertGreater(next_pending.sequence, pending.sequence)
        self.assertTrue(backend.drain_discard(next_pending))

    def test_kt_sync_failure_keeps_single_native_work_pending_until_drain(self):
        backend_module = importlib.import_module("round16.b.kt_expert_backend")
        routing = importlib.import_module("round16.b.routing_bridge")
        plan = routing.plan_routes(
            ((32, -1, -1, -1, -1, -1, -1, -1, -1, -1),),
            ((1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),),
            resident_logical_ids=tuple(range(32)),
        )

        class Wrapper:
            sync_calls = 0

            def submit_forward(self, *_args):
                pass

            def sync_forward(self, *_args):
                self.sync_calls += 1
                if self.sync_calls == 1:
                    raise RuntimeError("native FIFO has not drained yet")
                return ((1.0, 1.0),)

        wrapper = Wrapper()

        def stream_handle(_identity_key):
            return "stream"

        stream_handle.round16_api_identity = backend_module.KT_STREAM_HANDLE_API
        backend = backend_module.KTWeightedPartialBackend(
            wrapper,
            stream_handle_provider=stream_handle,
            api_identity=backend_module.KT_MOE_WRAPPER_FORWARD_API,
        )
        pending = backend.submit(plan, ((1.0,),), identity_key=("one",))
        with self.assertRaisesRegex(RuntimeError, "not drained"):
            backend.drain_discard(pending)
        with self.assertRaisesRegex(RuntimeError, "one outstanding"):
            backend.submit(plan, ((1.0,),), identity_key=("two",))
        self.assertTrue(backend.drain_discard(pending))
        replacement = backend.submit(plan, ((1.0,),), identity_key=("three",))
        self.assertTrue(backend.drain_discard(replacement))


if __name__ == "__main__":
    unittest.main()
