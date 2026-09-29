from __future__ import annotations

import importlib
from types import MappingProxyType, SimpleNamespace
import unittest


def _modules():
    return (
        importlib.import_module("round16.b.source_bound_hybrid_runner"),
        importlib.import_module("round16.b.selective_preallocation_installer"),
        importlib.import_module("round16.b.selective_loader"),
        importlib.import_module("round16.b.lifecycle_bridge"),
        importlib.import_module("round16.b.kt_expert_backend"),
    )


class _Router:
    def __init__(self, events, ids=None, on_select=None):
        self.events = events
        self.calls = 0
        raw_ids = ids or (0, 32, 1, 33, 2, 34, 3, 35, 4, 36)
        self.ids_rows = (
            tuple(tuple(row) for row in raw_ids)
            if raw_ids and isinstance(raw_ids[0], (tuple, list))
            else (tuple(raw_ids),)
        )
        self.last_hidden_states = None
        self.last_router_logits = None
        self.on_select = on_select

    def select_experts(self, **kwargs):
        self.calls += 1
        self.events.append("select")
        self.last_hidden_states = kwargs["hidden_states"]
        self.last_router_logits = kwargs["router_logits"]
        if self.on_select is not None:
            self.on_select()
        return (
            tuple((0.1,) * 10 for _row in self.ids_rows),
            self.ids_rows,
        )


class _QuantMethodMustNotRun:
    is_monolithic = False
    topk_indices_dtype = "int32"

    def apply(self, *_args, **_kwargs):
        raise AssertionError("Ascend quant apply must not run after selection")


class _RealShapedRoutedOwner:
    round16_q32_api_identity = "round16-b:routed-experts-q32-constructor:v1"

    def __init__(
        self,
        layer_name,
        params_dtype,
        moe_config,
        quant_config,
        expert_map_manager,
        *,
        round16_selective_plan,
        round16_resident_logical_ids,
        round16_logical_to_physical,
        round16_logical_expert_count,
        round16_physical_expert_count,
        **_kwargs,
    ):
        del params_dtype, moe_config, quant_config, expert_map_manager
        self.layer_name = layer_name
        self.round16_selective_plan = round16_selective_plan
        self.round16_resident_logical_ids = round16_resident_logical_ids
        self.round16_logical_to_physical = round16_logical_to_physical
        self.round16_logical_expert_count = round16_logical_expert_count
        self.round16_physical_expert_count = round16_physical_expert_count
        self.expert_map_manager = SimpleNamespace(
            map_global_to_local=lambda expert_id: round16_logical_to_physical[
                expert_id
            ]
        )
        self.w13_weight = SimpleNamespace(shape=(32, 8, 16), dtype="bfloat16")
        self.w2_weight = SimpleNamespace(shape=(32, 16, 8), dtype="bfloat16")
        self.quant_method = _QuantMethodMustNotRun()
        self._round16_executor = None

    def round16_bind_preselected_executor(self, executor, plan):
        if self._round16_executor is not None:
            raise RuntimeError("preselected executor already bound")
        if plan is not self.round16_selective_plan:
            raise RuntimeError("runner and routed owner plan identity differs")
        self._round16_executor = executor

    def forward_modular(
        self,
        *,
        x,
        topk_weights,
        topk_ids,
        shared_experts=None,
        shared_experts_input=None,
    ):
        del shared_experts, shared_experts_input
        if self._round16_executor is None:
            raise RuntimeError("preselected executor was not bound")
        return self._round16_executor(
            hidden_states=x,
            topk_ids=topk_ids,
            topk_weights=topk_weights,
            input_ids=None,
        )


class _ExactShapeUpstreamMoERunner:
    """CPU fake preserving the reviewed MoERunner owner boundaries."""

    def __init__(
        self,
        layer_name,
        moe_config,
        router,
        routed_experts,
        enable_dbo=False,
        gate=None,
        shared_experts=None,
        shared_expert_gate=None,
        routed_input_transform=None,
        routed_output_transform=None,
        routed_scaling_factor=1.0,
    ):
        del (
            moe_config,
            enable_dbo,
            gate,
            shared_expert_gate,
            routed_input_transform,
        )
        self.layer_name = layer_name
        self.router = router
        self.routed_experts = routed_experts
        self._shared_experts = shared_experts
        self.routed_output_transform = routed_output_transform
        self.routed_scaling_factor = routed_scaling_factor
        self.owner_events = router.events

    @property
    def is_internal_router(self):
        return False

    def __call__(self, *, hidden_states, router_logits, input_ids=None):
        return self.forward(hidden_states, router_logits, input_ids)

    def _apply_quant_method(
        self, hidden_states, router_logits, shared_experts_input, input_ids=None
    ):
        shared_output = None
        if self._shared_experts is not None:
            self.owner_events.append("upstream_shared")
            shared_output = self._shared_experts(shared_experts_input)
        topk_weights, topk_ids = self.router.select_experts(
            hidden_states=hidden_states,
            router_logits=router_logits,
            topk_indices_dtype=self.routed_experts.quant_method.topk_indices_dtype,
            input_ids=input_ids,
        )
        fused_output = self.routed_experts.forward_modular(
            x=hidden_states,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            shared_experts=self._shared_experts,
            shared_experts_input=shared_experts_input,
        )
        return shared_output, fused_output

    def forward(self, hidden_states, router_logits, input_ids=None):
        self.owner_events.append("upstream_forward")
        shared_output, fused_output = self._apply_quant_method(
            hidden_states, router_logits, hidden_states, input_ids
        )
        self.owner_events.append("upstream_scale")
        fused_output = tuple(
            tuple(value * self.routed_scaling_factor for value in row)
            for row in fused_output
        )
        self.owner_events.append("upstream_output_transform")
        fused_output = self.routed_output_transform(fused_output)
        self.owner_events.append("upstream_shared_add")
        result = tuple(
            tuple(left + right for left, right in zip(routed, shared))
            for routed, shared in zip(fused_output, shared_output)
        )
        self.owner_events.append("upstream_reduce")
        return tuple(tuple(value + 1000.0 for value in row) for row in result)


class _RealShapedQwenBlock:
    def __init__(self, qwen_module, prefix, *, router, shared, transform):
        self.events = router.events
        self.gate = lambda hidden: (
            tuple((0.0,) * 512 for _row in hidden),
            None,
        )
        self.experts = qwen_module.FusedMoE(
            prefix=f"{prefix}.experts",
            routed_scaling_factor=3.0,
            apply_routed_scale_to_output=True,
            shared_experts=shared,
            routed_output_transform=transform,
            router=router,
        )

    def forward(self, hidden_states):
        self.events.append("qwen_gate")
        router_logits, _ = self.gate(hidden_states)
        routed = self.experts(
            hidden_states=hidden_states,
            router_logits=router_logits,
        )
        self.events.append("qwen_residual")
        return tuple(
            tuple(value + 10000.0 for value in row) for row in routed
        )


class QwenRunnerWiringTests(unittest.TestCase):
    def _fixture(
        self,
        *,
        expected_layer_count=1,
        fail_prefix=None,
        native_result_factory=None,
        prepared_hidden_states=((50.0, 60.0),),
        prepared_router_logits=((1.0,) * 512,),
        prepared_valid_rows=(True,),
    ):
        runner, installer_module, loader, lifecycle_module, backend_module = (
            _modules()
        )
        events = []
        plan = loader.plan_q32(tuple(range(32)))
        lifecycle = lifecycle_module.RequestLeaseManager(slot_count=1)
        lease = lifecycle.acquire(
            "request-qwen",
            request_epoch=1,
            allocation_generation=1,
            batch_generation=1,
            layer_id=0,
            token_range=(0, 1),
            completion_id="completion-qwen",
            slot_id=0,
        )

        def unused_route_once(**_kwargs):
            raise AssertionError("top-level route_once callback must not run")

        unused_route_once.round16_api_identity = runner.SOURCE_BOUND_ROUTE_ONCE_API

        comm_method = object()

        def native_prepare(hidden_states, router_logits, layer_name):
            events.append("native_prepare")
            self.assertEqual(len(hidden_states), len(prepared_valid_rows))
            self.assertEqual(len(router_logits), len(prepared_valid_rows))
            self.assertTrue(layer_name.endswith(".experts"))
            return runner.NativeMoeCommPrepared(
                comm_method=comm_method,
                selector_hidden_states=prepared_hidden_states,
                selector_router_logits=prepared_router_logits,
                valid_rows=prepared_valid_rows,
                opaque_state="prepared-token",
                comm_type="ALLGATHER",
                ep_size=1,
                tp_size=1,
                quant_type="NONE",
                expert_map=None,
                log2phy=None,
                physical_expert_count=32,
            )

        native_prepare.round16_api_identity = (
            runner.SOURCE_BOUND_MOE_COMM_PREPARE_API
        )

        def current_comm_method():
            return current_comm_method.current

        current_comm_method.current = comm_method
        current_comm_method.round16_api_identity = (
            runner.SOURCE_BOUND_MOE_COMM_CONTEXT_API
        )

        def native_weighted_partial(
            prepared, route_plan, hidden_states, _identity_key
        ):
            events.append("native")
            self.assertIs(prepared.comm_method, comm_method)
            self.assertEqual(prepared.opaque_state, "prepared-token")
            self.assertEqual(hidden_states, prepared.selector_hidden_states)
            for row_index, valid in enumerate(prepared.valid_rows):
                if not valid:
                    self.assertEqual(route_plan.routes.topk_ids[row_index], (0,) * 10)
                    self.assertEqual(route_plan.routes.topk_weights[row_index], (0.0,) * 10)
            if native_result_factory is not None:
                return native_result_factory(runner)
            npu_tokens = {route.token_index for route in route_plan.npu_routes}
            return runner.NativeWeightedPartial(
                tuple(
                    (0.5, 1.0) if row in npu_tokens else (0.0, 0.0)
                    for row in range(route_plan.routes.token_count)
                )
            )

        native_weighted_partial.round16_api_identity = (
            runner.SOURCE_BOUND_NATIVE_WEIGHTED_PARTIAL_API
        )

        class Wrapper:
            def submit_forward(self, _hidden, _ids, weights, _stream):
                events.append("cpu_submit")
                self.weights = tuple(tuple(row) for row in weights)

            def sync_forward(self, *_args):
                events.append("cpu_sync")
                return tuple(
                    (1.5, 2.0) if any(row) else (0.0, 0.0)
                    for row in self.weights
                )

        def stream_handle(_identity_key):
            return "stream-qwen"

        stream_handle.round16_api_identity = backend_module.KT_STREAM_HANDLE_API
        backend = backend_module.KTWeightedPartialBackend(
            Wrapper(),
            stream_handle_provider=stream_handle,
            api_identity=backend_module.KT_MOE_WRAPPER_FORWARD_API,
        )

        def context_provider(layer_name):
            self.assertTrue(layer_name.endswith(".experts"))
            return runner.SourceBoundForwardContext(
                lease=lease,
                input_fingerprint="qwen-input-1",
            )

        context_provider.round16_api_identity = runner.SOURCE_BOUND_FORWARD_CONTEXT_API
        dependencies = runner.SourceBoundRunnerDependencies(
            lifecycle=lifecycle,
            route_once=unused_route_once,
            cpu_backend=backend,
            forward_context_provider=context_provider,
            native_moe_comm_prepare=native_prepare,
            moe_comm_method_provider=current_comm_method,
            native_weighted_partial_execute=native_weighted_partial,
        )
        native_runner_cls = runner.build_source_bound_native_runner_class(
            _ExactShapeUpstreamMoERunner,
            upstream_api_identity=runner.UPSTREAM_MOE_RUNNER_SOURCE_API,
        )
        self.assertIsNone(dependencies.native_preselected_execute)

        created = []

        def ascend_fused_moe(*_args, **kwargs):
            prefix = kwargs["prefix"]
            if fail_prefix is not None and prefix == fail_prefix:
                raise RuntimeError("real-shaped Qwen construction failed")
            routed_args = kwargs["routed_experts_args"]
            runner_args = kwargs["runner_args"]
            self.assertIsInstance(routed_args, MappingProxyType)
            self.assertIsInstance(runner_args, MappingProxyType)
            owner = kwargs["routed_experts_cls"](
                prefix,
                "bfloat16",
                object(),
                None,
                object(),
                **dict(routed_args),
            )
            instance = kwargs["runner_cls"](
                layer_name=prefix,
                moe_config=object(),
                router=kwargs.get("router") or _Router(events),
                routed_experts=owner,
                shared_experts=kwargs.get("shared_experts"),
                routed_output_transform=kwargs.get("routed_output_transform"),
                routed_scaling_factor=kwargs["routed_scaling_factor"],
                **dict(runner_args),
            )
            created.append(instance)
            return instance

        ascend_fused_moe.__name__ = "_ascend_FusedMoE"
        qwen_module = SimpleNamespace(
            __name__="vllm.model_executor.models.qwen3_next",
            FusedMoE=ascend_fused_moe,
        )
        installer = installer_module.ModelScopedSelectivePreallocationInstaller(
            qwen_module=qwen_module,
            ascend_fused_moe=ascend_fused_moe,
            ascend_api_identity=installer_module.ASCEND_FUSED_MOE_FORWARDING_API,
            routed_experts_cls=_RealShapedRoutedOwner,
            routed_experts_api_identity=installer_module.ROUTED_EXPERTS_Q32_API,
            hybrid_runner_cls=native_runner_cls,
            hybrid_runner_api_identity=runner.SOURCE_BOUND_HYBRID_RUNNER_API,
            hybrid_runner_args={"round16_runner_dependencies": dependencies},
            expected_layer_count=expected_layer_count,
        )
        return (
            runner,
            plan,
            lifecycle,
            lease,
            events,
            qwen_module,
            ascend_fused_moe,
            installer,
            created,
        )

    def test_qwen_self_experts_forward_routes_once_and_upstream_owns_postprocess(self):
        (
            _runner,
            plan,
            lifecycle,
            lease,
            events,
            qwen_module,
            _original,
            installer,
            created,
        ) = self._fixture()
        router = _Router(events)

        def shared(_hidden):
            return ((10.0, 20.0),)

        def transform(rows):
            return tuple((row[0] + 100.0, row[1] + 200.0) for row in rows)

        with installer.activate(
            plan=plan,
            architecture="Qwen4ExpForConditionalGeneration",
            model_class=object,
        ) as acknowledge:
            block = _RealShapedQwenBlock(
                qwen_module,
                "model.layers.0.mlp",
                router=router,
                shared=shared,
                transform=transform,
            )
            self.assertTrue(acknowledge(object()))
            output = block.forward(((5.0, 6.0),))

        self.assertEqual(output, ((11116.0, 11229.0),))
        self.assertEqual(router.calls, 1)
        self.assertEqual(router.last_hidden_states, ((50.0, 60.0),))
        self.assertEqual(router.last_router_logits, ((1.0,) * 512,))
        self.assertEqual(
            events,
            [
                "qwen_gate",
                "upstream_forward",
                "native_prepare",
                "upstream_shared",
                "select",
                "cpu_submit",
                "native",
                "cpu_sync",
                "upstream_scale",
                "upstream_output_transform",
                "upstream_shared_add",
                "upstream_reduce",
                "qwen_residual",
            ],
        )
        self.assertIs(block.experts, created[0])
        self.assertIs(block.experts.round16_selective_plan, plan)
        self.assertIs(
            block.experts.routed_experts.round16_selective_plan,
            plan,
        )
        self.assertEqual(len(lifecycle.drain(lease)), 1)

    def test_48_layer_ack_revalidates_runner_owner_plan_identity(self):
        (
            _runner,
            plan,
            _lifecycle,
            _lease,
            events,
            qwen_module,
            _original,
            installer,
            created,
        ) = self._fixture(expected_layer_count=48)
        with installer.activate(
            plan=plan,
            architecture="Qwen4ExpForConditionalGeneration",
            model_class=object,
        ) as acknowledge:
            for layer in range(48):
                qwen_module.FusedMoE(
                    prefix=f"model.layers.{layer}.mlp.experts",
                    router=_Router(events),
                    shared_experts=None,
                    routed_output_transform=lambda value: value,
                    routed_scaling_factor=1.0,
                )
            self.assertEqual(len(created), 48)
            self.assertTrue(acknowledge(object()))
        for item in created:
            self.assertIs(item.round16_selective_plan, plan)
            self.assertIs(item.routed_experts.round16_selective_plan, plan)
            self.assertTrue(item.round16_runner_owner_binding_verified)

    def test_real_shaped_construction_exception_restores_qwen_binding(self):
        (
            _runner,
            plan,
            _lifecycle,
            _lease,
            events,
            qwen_module,
            original,
            installer,
            created,
        ) = self._fixture(
            expected_layer_count=48,
            fail_prefix="model.layers.17.mlp.experts",
        )
        with self.assertRaisesRegex(RuntimeError, "construction failed"):
            with installer.activate(
                plan=plan,
                architecture="Qwen4ExpForConditionalGeneration",
                model_class=object,
            ):
                for layer in range(48):
                    qwen_module.FusedMoE(
                        prefix=f"model.layers.{layer}.mlp.experts",
                        router=_Router(events),
                        shared_experts=None,
                        routed_output_transform=lambda value: value,
                        routed_scaling_factor=1.0,
                    )
        self.assertEqual(len(created), 17)
        self.assertIs(qwen_module.FusedMoE, original)

    def test_native_boundary_rejects_occurrence_mapping_and_drains_cpu(self):
        (
            _runner,
            plan,
            lifecycle,
            lease,
            events,
            qwen_module,
            _original,
            installer,
            _created,
        ) = self._fixture(
            native_result_factory=lambda _runner: {(0, 0, 0): (0.5, 1.0)}
        )
        with installer.activate(
            plan=plan,
            architecture="Qwen4ExpForConditionalGeneration",
            model_class=object,
        ) as acknowledge:
            block = _RealShapedQwenBlock(
                qwen_module,
                "model.layers.0.mlp",
                router=_Router(events),
                shared=lambda _hidden: ((10.0, 20.0),),
                transform=lambda rows: rows,
            )
            self.assertTrue(acknowledge(object()))
            with self.assertRaisesRegex(TypeError, "NativeWeightedPartial"):
                block.forward(((5.0, 6.0),))
        self.assertIn("cpu_sync", events)
        self.assertEqual(lifecycle.drain(lease), ())

    def test_native_boundary_rejects_unfinalized_aggregate(self):
        with self.assertRaisesRegex(ValueError, "prepare/finalize"):
            self._fixture()[0].NativeWeightedPartial(
                ((0.5, 1.0),), finalize_applied=False
            )

    def test_all_cpu_and_all_npu_still_use_prepare_finalize_envelope(self):
        cases = (
            (tuple(range(32, 42)), True),
            (tuple(range(10)), False),
        )
        for ids, expect_cpu in cases:
            with self.subTest(expect_cpu=expect_cpu):
                (
                    _runner,
                    plan,
                    _lifecycle,
                    _lease,
                    events,
                    qwen_module,
                    _original,
                    installer,
                    _created,
                ) = self._fixture()
                router = _Router(events, ids=ids)
                with installer.activate(
                    plan=plan,
                    architecture="Qwen4ExpForConditionalGeneration",
                    model_class=object,
                ) as acknowledge:
                    block = _RealShapedQwenBlock(
                        qwen_module,
                        "model.layers.0.mlp",
                        router=router,
                        shared=lambda _hidden: ((0.0, 0.0),),
                        transform=lambda rows: rows,
                    )
                    self.assertTrue(acknowledge(object()))
                    block.forward(((5.0, 6.0),))
                self.assertEqual(events.count("native_prepare"), 1)
                self.assertEqual(events.count("select"), 1)
                self.assertEqual(events.count("native"), 1)
                self.assertEqual("cpu_submit" in events, expect_cpu)
                self.assertEqual("cpu_sync" in events, expect_cpu)

    def test_comm_context_identity_change_after_prepare_fails_closed(self):
        (
            _runner,
            plan,
            lifecycle,
            lease,
            events,
            qwen_module,
            _original,
            installer,
            created,
        ) = self._fixture()
        with installer.activate(
            plan=plan,
            architecture="Qwen4ExpForConditionalGeneration",
            model_class=object,
        ) as acknowledge:
            block = _RealShapedQwenBlock(
                qwen_module,
                "model.layers.0.mlp",
                router=_Router(events),
                shared=lambda _hidden: ((0.0, 0.0),),
                transform=lambda rows: rows,
            )
            self.assertTrue(acknowledge(object()))
            provider = created[0].round16_preselected_executor.dependencies.moe_comm_method_provider
            block.experts.router.on_select = lambda: setattr(
                provider, "current", object()
            )
            with self.assertRaisesRegex(RuntimeError, "MoEComm object changed"):
                block.forward(((5.0, 6.0),))
        self.assertNotIn("cpu_submit", events)
        self.assertEqual(lifecycle.drain(lease), ())

    def test_prepared_invalid_row_is_zeroed_before_both_branch_payloads(self):
        (
            _runner,
            plan,
            _lifecycle,
            _lease,
            events,
            qwen_module,
            _original,
            installer,
            _created,
        ) = self._fixture(
            prepared_hidden_states=((50.0, 60.0), (70.0, 80.0)),
            prepared_router_logits=((1.0,) * 512, (2.0,) * 512),
            prepared_valid_rows=(True, False),
        )
        router = _Router(
            events,
            ids=(
                (0, 32, 1, 33, 2, 34, 3, 35, 4, 36),
                (40, 41, 42, 43, 44, 45, 46, 47, 48, 49),
            ),
        )
        with installer.activate(
            plan=plan,
            architecture="Qwen4ExpForConditionalGeneration",
            model_class=object,
        ) as acknowledge:
            block = _RealShapedQwenBlock(
                qwen_module,
                "model.layers.0.mlp",
                router=router,
                shared=lambda _hidden: ((0.0, 0.0), (0.0, 0.0)),
                transform=lambda rows: rows,
            )
            self.assertTrue(acknowledge(object()))
            output = block.forward(((5.0, 6.0), (7.0, 8.0)))
        self.assertEqual(output[1], (11000.0, 11000.0))
        self.assertEqual(events.count("select"), 1)
        self.assertEqual(events.count("native"), 1)


if __name__ == "__main__":
    unittest.main()
