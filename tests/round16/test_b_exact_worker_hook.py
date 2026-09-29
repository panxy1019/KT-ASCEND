from __future__ import annotations

from contextlib import contextmanager
import importlib
from types import SimpleNamespace
import unittest


def _modules():
    return (
        importlib.import_module("round16.b.exact_donor_worker_hook"),
        importlib.import_module("round16.b.selective_loader"),
        importlib.import_module("round16.b.source_bound_hybrid_runner"),
    )


class _Weight:
    def __init__(self, shape):
        self.shape = shape
        self.dtype = "bfloat16"


class _Owner:
    round16_q32_api_identity = "round16-b:routed-experts-q32-constructor:v1"

    def __init__(self, **kwargs):
        plan = kwargs["round16_selective_plan"]
        self.round16_selective_plan = plan
        self.round16_resident_logical_ids = kwargs[
            "round16_resident_logical_ids"
        ]
        self.round16_logical_to_physical = kwargs[
            "round16_logical_to_physical"
        ]
        self.round16_logical_expert_count = kwargs[
            "round16_logical_expert_count"
        ]
        self.round16_physical_expert_count = kwargs[
            "round16_physical_expert_count"
        ]
        self.expert_map_manager = SimpleNamespace(
            global_num_experts=512,
            local_num_experts=32,
            map_global_to_local=lambda logical_id: plan.logical_to_physical[
                logical_id
            ],
        )
        self.w13_weight = _Weight((32, 2560, 1280))
        self.w2_weight = _Weight((32, 640, 2560))


class _ConstructedRunner:
    round16_source_bound_runner_api_identity = (
        "round16-b:source-bound-hybrid-runner:offline-contract-v1"
    )

    def __init__(
        self,
        *,
        routed_experts,
        round16_selective_plan,
        round16_runtime_registry,
        **_kwargs,
    ):
        self.layer_name = routed_experts.layer_name
        self.routed_experts = routed_experts
        self.round16_routed_owner = routed_experts
        self.round16_selective_plan = round16_selective_plan
        self.round16_runtime_registry = round16_runtime_registry
        self.round16_runner_owner_binding_verified = True


class _FakeLayerRuntime:
    round16_api_identity = "round16-b:exact-worker-layer-runtime:v1"

    def __init__(self, layer_name, runner, events):
        self.layer_name = layer_name
        self.runner = runner
        self.routed_owner = runner.routed_experts
        self.events = events

    def prepare(self, *_args, **_kwargs):
        raise AssertionError("post-load registration must not run forward")

    def execute_selected(self, *_args, **_kwargs):
        raise AssertionError("post-load registration must not run forward")

    def abort_prepared(self, *_args, **_kwargs):
        raise AssertionError("post-load registration must not run forward")


def _source_token(module):
    return module.validate_exact_worker_source_identity(
        dict(module.EXACT_WORKER_SOURCE_SHA256)
    )


def _worker_fixture(*, bad_owner=False):
    hook, loader, _runner_module = _modules()
    installer_module = importlib.import_module(
        "round16.b.selective_preallocation_installer"
    )
    plan = loader.plan_q32(tuple(range(32)))
    events = []

    def ascend_fused_moe(*_args, **kwargs):
        events.append(("construct", kwargs["prefix"]))
        owner = kwargs["routed_experts_cls"](**dict(kwargs["routed_experts_args"]))
        owner.layer_name = kwargs["prefix"]
        if bad_owner:
            owner.w13_weight.shape = (512, 2560, 1280)
        runner_args = dict(kwargs["runner_args"])
        return kwargs["runner_cls"](
            routed_experts=owner,
            **runner_args,
        )

    ascend_fused_moe.__name__ = "_ascend_FusedMoE"
    qwen_module = SimpleNamespace(
        __name__="vllm.model_executor.models.qwen3_next",
        FusedMoE=ascend_fused_moe,
    )
    registry = hook.ExactWorkerRuntimeRegistry(
        source_token=_source_token(hook),
        plan=plan,
        expected_layer_count=48,
    )
    installer = installer_module.ModelScopedSelectivePreallocationInstaller(
        qwen_module=qwen_module,
        ascend_fused_moe=ascend_fused_moe,
        ascend_api_identity=installer_module.ASCEND_FUSED_MOE_FORWARDING_API,
        routed_experts_cls=_Owner,
        routed_experts_api_identity=installer_module.ROUTED_EXPERTS_Q32_API,
        hybrid_runner_cls=_ConstructedRunner,
        hybrid_runner_api_identity=(
            installer_module.SOURCE_BOUND_HYBRID_RUNNER_API
        ),
        hybrid_runner_args={"round16_runtime_registry": registry},
        expected_layer_count=48,
    )

    class NPUModelRunner:
        def __init__(self):
            self.model_config = SimpleNamespace(
                hf_config=SimpleNamespace(
                    architectures=["Qwen4ExpForConditionalGeneration"]
                )
            )
            self.model = None

        def load_model(self):
            events.append("load-start")
            layers = []
            for index in range(48):
                expert = qwen_module.FusedMoE(
                    prefix=f"language_model.model.layers.{index}.mlp.experts"
                )
                layers.append(SimpleNamespace(mlp=SimpleNamespace(experts=expert)))
            self.model = SimpleNamespace(
                language_model=SimpleNamespace(
                    model=SimpleNamespace(layers=layers)
                )
            )
            events.append("load-complete")

        def get_model(self):
            return self.model

    NPUModelRunner.__module__ = "vllm_ascend.worker.model_runner_v1"

    def runtime_factory(*, layer_name, runner, source_token, plan):
        events.append(("runtime", layer_name))
        if source_token != registry.source_token or plan is not registry.plan:
            raise AssertionError("factory identity binding changed")
        return _FakeLayerRuntime(layer_name, runner, events)

    runtime_factory.round16_api_identity = hook.EXACT_LAYER_RUNTIME_FACTORY_API
    return (
        hook,
        plan,
        events,
        qwen_module,
        ascend_fused_moe,
        registry,
        installer,
        NPUModelRunner,
        runtime_factory,
    )


class ExactWorkerSourceTests(unittest.TestCase):
    def test_unknown_source_hash_rejected_before_patch(self):
        hook, *_rest = _worker_fixture()
        observed = dict(hook.EXACT_WORKER_SOURCE_SHA256)
        observed["model_runner_v1.py"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "source identity"):
            hook.validate_exact_worker_source_identity(observed)

    def test_default_off_does_not_patch_or_construct(self):
        (
            hook,
            plan,
            events,
            _qwen,
            _ascend,
            registry,
            installer,
            runner_cls,
            factory,
        ) = _worker_fixture()
        original = runner_cls.load_model
        handle = hook.install_exact_worker_load_hook(
            model_runner_cls=runner_cls,
            source_token=registry.source_token,
            installer=installer,
            plan=plan,
            runtime_registry=registry,
            runtime_factory=factory,
            enabled=False,
        )
        self.assertFalse(handle.installed)
        self.assertIs(runner_cls.load_model, original)
        self.assertEqual(events, [])


class ExactWorkerHookTests(unittest.TestCase):
    def test_load_wraps_preallocation_then_atomically_registers_48_layers(self):
        (
            hook,
            plan,
            events,
            qwen,
            original_fused,
            registry,
            installer,
            runner_cls,
            factory,
        ) = _worker_fixture()
        original_load = runner_cls.load_model
        handle = hook.install_exact_worker_load_hook(
            model_runner_cls=runner_cls,
            source_token=registry.source_token,
            installer=installer,
            plan=plan,
            runtime_registry=registry,
            runtime_factory=factory,
            enabled=True,
        )
        try:
            worker_runner = runner_cls()
            worker_runner.load_model()
            self.assertIs(qwen.FusedMoE, original_fused)
            self.assertEqual(registry.layer_count, 48)
            self.assertIs(worker_runner.round16_b_runtime_registry, registry)
            self.assertEqual(events[0], "load-start")
            self.assertEqual(events[1], ("construct", "language_model.model.layers.0.mlp.experts"))
            self.assertIn("load-complete", events)
            first_runtime = registry.require(
                "language_model.model.layers.0.mlp.experts",
                worker_runner.model.language_model.model.layers[0].mlp.experts,
            )
            self.assertIs(first_runtime.routed_owner.w13_weight.shape[0], 32)
        finally:
            handle.uninstall()
        self.assertIs(runner_cls.load_model, original_load)

    def test_bad_preallocation_fails_before_registration_and_restores_qwen(self):
        (
            hook,
            plan,
            _events,
            qwen,
            original_fused,
            registry,
            installer,
            runner_cls,
            factory,
        ) = _worker_fixture(bad_owner=True)
        handle = hook.install_exact_worker_load_hook(
            model_runner_cls=runner_cls,
            source_token=registry.source_token,
            installer=installer,
            plan=plan,
            runtime_registry=registry,
            runtime_factory=factory,
            enabled=True,
        )
        try:
            with self.assertRaisesRegex(RuntimeError, "physical expert"):
                runner_cls().load_model()
            self.assertEqual(registry.layer_count, 0)
            self.assertIs(qwen.FusedMoE, original_fused)
        finally:
            handle.uninstall()


class _PreparedRuntime:
    round16_api_identity = "round16-b:exact-worker-layer-runtime:v1"

    def __init__(self, runner_module, events):
        self.module = runner_module
        self.events = events
        self.layer_name = "language_model.model.layers.0.mlp.experts"
        self.runner = None
        self.routed_owner = None
        self.prepared_hidden = object()
        self.prepared_router = object()

    def prepare(self, hidden_states, router_logits, layer_name):
        self.events.append(("prepare", hidden_states, router_logits, layer_name))
        return self.module.NativeMoeCommPrepared(
            comm_method=object(),
            selector_hidden_states=self.prepared_hidden,
            selector_router_logits=self.prepared_router,
            valid_rows=(True,),
            opaque_state=object(),
            comm_type="ALLGATHER",
            ep_size=1,
            tp_size=1,
            quant_type="NONE",
            expert_map=None,
            log2phy=None,
            physical_expert_count=32,
        )

    def execute_selected(self, prepared, selected, hidden_states, layer_name):
        self.events.extend(("cpu-submit", "native-preselected", "cpu-sync"))
        if hidden_states is not prepared.selector_hidden_states:
            raise AssertionError("prepared hidden identity changed")
        if layer_name != self.layer_name or selected.valid_rows != (True,):
            raise AssertionError("selected runtime identity changed")
        return ((2.0, 4.0),)

    def cleanup(self, prepared):
        self.events.append(("cleanup", prepared))

    def abort_prepared(self, prepared):
        self.cleanup(prepared)


class _ExactSignatureUpstreamRunner:
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
            routed_output_transform,
        )
        self.layer_name = layer_name
        self.router = router
        self.routed_experts = routed_experts
        self._shared_experts = shared_experts
        self.routed_scaling_factor = routed_scaling_factor

    @property
    def _quant_method(self):
        return self.routed_experts.quant_method

    def _maybe_apply_shared_experts(self, hidden_states, order):
        self.router.events.append(("shared", order))
        if self._shared_experts is not None:
            self._shared_experts.output = self._shared_experts(hidden_states)

    def forward(self, hidden_states, router_logits, input_ids=None):
        shared, routed = self._apply_quant_method(
            hidden_states, router_logits, hidden_states, input_ids
        )
        routed = tuple(
            tuple(value * self.routed_scaling_factor for value in row)
            for row in routed
        )
        return tuple(
            tuple(a + b for a, b in zip(route_row, shared_row))
            for route_row, shared_row in zip(routed, shared)
        )


class ExactWorkerForwardTests(unittest.TestCase):
    def test_fake_exact_runner_selects_once_and_preserves_owner_boundaries(self):
        hook, loader, runner_module = _modules()
        plan = loader.plan_q32(tuple(range(32)))
        events = []

        class Router:
            def __init__(self):
                self.calls = 0
                self.events = events

            def select_experts(self, **kwargs):
                self.calls += 1
                events.append(("select", kwargs["hidden_states"], kwargs["router_logits"]))
                return object(), object()

        class Shared:
            output = None

            def __call__(self, _hidden):
                return ((5.0, 7.0),)

        owner = SimpleNamespace(
            round16_selective_plan=plan,
            quant_method=SimpleNamespace(topk_indices_dtype="int32"),
        )
        registry = hook.ExactWorkerRuntimeRegistry(
            source_token=_source_token(hook),
            plan=plan,
            expected_layer_count=1,
        )
        runtime = _PreparedRuntime(runner_module, events)
        router = Router()
        generated = hook.build_exact_worker_runner_class(
            _ExactSignatureUpstreamRunner,
            upstream_api_identity=hook.EXACT_UPSTREAM_MOE_RUNNER_API,
            shared_order_no_overlap="NO_OVERLAP",
            shared_order_overlap="MULTI_STREAM_OVERLAPPED",
        )
        runner = generated(
            layer_name=runtime.layer_name,
            moe_config=object(),
            router=router,
            routed_experts=owner,
            shared_experts=Shared(),
            routed_scaling_factor=3.0,
            round16_selective_plan=plan,
            round16_runtime_registry=registry,
        )
        runtime.runner = runner
        runtime.routed_owner = owner
        registry.register_prepared(((runtime.layer_name, runner, runtime),))

        routed_plus_shared = runner.forward(object(), object())
        with_residual = tuple(
            tuple(value + 100.0 for value in row) for row in routed_plus_shared
        )
        self.assertEqual(router.calls, 1)
        self.assertEqual(routed_plus_shared, ((11.0, 19.0),))
        self.assertEqual(with_residual, ((111.0, 119.0),))
        self.assertEqual(events[1][0], "shared")
        self.assertEqual(events[2][0], "select")
        self.assertEqual(events[3:6], ["cpu-submit", "native-preselected", "cpu-sync"])
        self.assertEqual(events[-1], ("shared", "MULTI_STREAM_OVERLAPPED"))

    def test_device_selector_replaces_cuda_router_without_changing_merge(self):
        hook, loader, runner_module = _modules()
        plan = loader.plan_q32(tuple(range(32)))
        events = []

        class Router:
            def __init__(self):
                self.calls = 0
                self.events = events

            def select_experts(self, **_kwargs):
                self.calls += 1
                raise AssertionError("device selector must bypass CUDA router")

        class Shared:
            output = None

            def __call__(self, _hidden):
                return ((5.0, 7.0),)

        def device_selector(**kwargs):
            events.append(
                (
                    "device-select",
                    kwargs["hidden_states"],
                    kwargs["router_logits"],
                    kwargs["input_ids"],
                )
            )
            return object(), object()

        device_selector.round16_api_identity = (
            hook.EXACT_WORKER_EXPERT_SELECTOR_API
        )
        owner = SimpleNamespace(
            round16_selective_plan=plan,
            quant_method=SimpleNamespace(topk_indices_dtype="int32"),
        )
        registry = hook.ExactWorkerRuntimeRegistry(
            source_token=_source_token(hook),
            plan=plan,
            expected_layer_count=1,
        )
        runtime = _PreparedRuntime(runner_module, events)
        router = Router()
        generated = hook.build_exact_worker_runner_class(
            _ExactSignatureUpstreamRunner,
            upstream_api_identity=hook.EXACT_UPSTREAM_MOE_RUNNER_API,
            shared_order_no_overlap="NO_OVERLAP",
            shared_order_overlap="MULTI_STREAM_OVERLAPPED",
            expert_selector=device_selector,
        )
        runner = generated(
            layer_name=runtime.layer_name,
            moe_config=object(),
            router=router,
            routed_experts=owner,
            shared_experts=Shared(),
            routed_scaling_factor=3.0,
            round16_selective_plan=plan,
            round16_runtime_registry=registry,
        )
        runtime.runner = runner
        runtime.routed_owner = owner
        registry.register_prepared(((runtime.layer_name, runner, runtime),))

        token_ids = object()
        result = runner.forward(object(), object(), token_ids)
        self.assertEqual(router.calls, 0)
        self.assertEqual(result, ((11.0, 19.0),))
        selected = [event for event in events if event[0] == "device-select"]
        self.assertEqual(len(selected), 1)
        self.assertIs(selected[0][3], token_ids)
        self.assertEqual(
            generated.router_owner,
            "ASCEND_EXPERT_SELECTOR_ON_PREPARED_INPUTS",
        )

    def test_forward_before_post_load_registration_fails_closed(self):
        hook, loader, _runner_module = _modules()
        plan = loader.plan_q32(tuple(range(32)))
        registry = hook.ExactWorkerRuntimeRegistry(
            source_token=_source_token(hook), plan=plan, expected_layer_count=1
        )
        generated = hook.build_exact_worker_runner_class(
            _ExactSignatureUpstreamRunner,
            upstream_api_identity=hook.EXACT_UPSTREAM_MOE_RUNNER_API,
            shared_order_no_overlap="before",
            shared_order_overlap="after",
        )
        router = SimpleNamespace(events=[], select_experts=lambda **_kwargs: (object(), object()))
        runner = generated(
            layer_name="language_model.model.layers.0.mlp.experts",
            moe_config=object(),
            router=router,
            routed_experts=SimpleNamespace(
                round16_selective_plan=plan,
                quant_method=SimpleNamespace(topk_indices_dtype="int32"),
            ),
            round16_selective_plan=plan,
            round16_runtime_registry=registry,
        )
        with self.assertRaisesRegex(RuntimeError, "not registered"):
            runner.forward(object(), object())


if __name__ == "__main__":
    unittest.main()
