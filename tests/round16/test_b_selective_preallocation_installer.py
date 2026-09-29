from __future__ import annotations

import importlib
from types import MappingProxyType, SimpleNamespace
import unittest

from round16.b.selective_loader import plan_q32


def _installer_module():
    try:
        return importlib.import_module("round16.b.selective_preallocation_installer")
    except ModuleNotFoundError as error:
        raise AssertionError(
            "round16.b.selective_preallocation_installer must exist"
        ) from error


class _VerifiedRoutedExperts:
    round16_q32_api_identity = "round16-b:routed-experts-q32-constructor:v1"

    def __init__(
        self,
        *,
        round16_selective_plan,
        round16_resident_logical_ids,
        round16_logical_to_physical,
        round16_logical_expert_count,
        round16_physical_expert_count,
    ) -> None:
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


class _Runner:
    def __init__(self, routed_experts) -> None:
        self.routed_experts = routed_experts


def _fixture(module, *, expected_layer_count=2):
    calls: list[dict[str, object]] = []

    def ascend_fused_moe(*args, **kwargs):
        del args
        calls.append(dict(kwargs))
        routed_args = kwargs["routed_experts_args"]
        if not isinstance(routed_args, MappingProxyType):
            raise AssertionError("Q32 routed args must be immutable")
        routed_class = kwargs["routed_experts_cls"]
        return _Runner(routed_class(**dict(routed_args)))

    ascend_fused_moe.__name__ = "_ascend_FusedMoE"
    qwen_module = SimpleNamespace(
        __name__="vllm.model_executor.models.qwen3_next",
        FusedMoE=ascend_fused_moe,
    )
    installer = module.ModelScopedSelectivePreallocationInstaller(
        qwen_module=qwen_module,
        ascend_fused_moe=ascend_fused_moe,
        ascend_api_identity=module.ASCEND_FUSED_MOE_FORWARDING_API,
        routed_experts_cls=_VerifiedRoutedExperts,
        routed_experts_api_identity=module.ROUTED_EXPERTS_Q32_API,
        expected_layer_count=expected_layer_count,
    )
    return installer, qwen_module, ascend_fused_moe, calls


class ModelScopedSelectivePreallocationInstallerTests(unittest.TestCase):
    def test_context_injects_same_immutable_q32_plan_for_every_layer_and_restores(self):
        module = _installer_module()
        installer, qwen_module, original, calls = _fixture(module)
        plan = plan_q32(tuple(range(32)))

        with installer.activate(
            plan=plan,
            architecture="Qwen4ExpForConditionalGeneration",
            model_class=object,
        ) as acknowledge:
            first = qwen_module.FusedMoE(prefix="model.layers.0.mlp.experts")
            second = qwen_module.FusedMoE(prefix="model.layers.1.mlp.experts")
            self.assertTrue(acknowledge(object()))

        self.assertIs(qwen_module.FusedMoE, original)
        self.assertEqual(len(calls), 2)
        for call, runner in zip(calls, (first, second)):
            self.assertIs(call["routed_experts_cls"], _VerifiedRoutedExperts)
            self.assertIsInstance(call["routed_experts_args"], MappingProxyType)
            self.assertIs(runner.routed_experts.round16_selective_plan, plan)
            self.assertEqual(
                runner.routed_experts.round16_logical_to_physical,
                plan.logical_to_physical,
            )

    def test_exception_and_ack_failure_restore_original_binding(self):
        module = _installer_module()
        installer, qwen_module, original, _calls = _fixture(
            module, expected_layer_count=2
        )
        plan = plan_q32(tuple(range(32)))

        with self.assertRaisesRegex(RuntimeError, "construction failed"):
            with installer.activate(
                plan=plan,
                architecture="Qwen4ExpForConditionalGeneration",
                model_class=object,
            ):
                qwen_module.FusedMoE(prefix="layer.0")
                raise RuntimeError("construction failed")
        self.assertIs(qwen_module.FusedMoE, original)

        with self.assertRaisesRegex(RuntimeError, "expected 2"):
            with installer.activate(
                plan=plan,
                architecture="Qwen4ExpForConditionalGeneration",
                model_class=object,
            ) as acknowledge:
                qwen_module.FusedMoE(prefix="layer.0")
                acknowledge(object())
        self.assertIs(qwen_module.FusedMoE, original)

    def test_conflicting_routed_expert_override_fails_before_original_call(self):
        module = _installer_module()
        installer, qwen_module, original, calls = _fixture(
            module, expected_layer_count=1
        )
        plan = plan_q32(tuple(range(32)))

        with installer.activate(
            plan=plan,
            architecture="Qwen4ExpForConditionalGeneration",
            model_class=object,
        ):
            with self.assertRaisesRegex(RuntimeError, "already supplies"):
                qwen_module.FusedMoE(
                    routed_experts_cls=object,
                    routed_experts_args={"unsafe": True},
                )

        self.assertEqual(calls, [])
        self.assertIs(qwen_module.FusedMoE, original)

    def test_tampered_layer_mapping_fails_ack_and_restores(self):
        module = _installer_module()
        installer, qwen_module, original, _calls = _fixture(
            module, expected_layer_count=1
        )
        plan = plan_q32(tuple(range(32)))

        with self.assertRaisesRegex(RuntimeError, "mapping"):
            with installer.activate(
                plan=plan,
                architecture="Qwen4ExpForConditionalGeneration",
                model_class=object,
            ) as acknowledge:
                runner = qwen_module.FusedMoE(prefix="layer.0")
                runner.routed_experts.round16_logical_to_physical = tuple(
                    reversed(plan.logical_to_physical)
                )
                acknowledge(object())

        self.assertIs(qwen_module.FusedMoE, original)

    def test_ack_revalidates_live_mapper_q32_shapes_and_bfloat16_dtype(self):
        module = _installer_module()
        plan = plan_q32(tuple(range(32)))
        for name, tamper, message in (
            (
                "mapper",
                lambda owner: setattr(
                    owner.expert_map_manager,
                    "map_global_to_local",
                    lambda expert_id: 1 if expert_id == 0 else plan.logical_to_physical[expert_id],
                ),
                "mapper",
            ),
            (
                "shape",
                lambda owner: setattr(owner.w13_weight, "shape", (31, 8, 16)),
                "physical expert dimension",
            ),
            (
                "dtype",
                lambda owner: setattr(owner.w2_weight, "dtype", "float16"),
                "bfloat16",
            ),
        ):
            with self.subTest(name=name):
                installer, qwen_module, original, _calls = _fixture(
                    module, expected_layer_count=1
                )
                with self.assertRaisesRegex(RuntimeError, message):
                    with installer.activate(
                        plan=plan,
                        architecture="Qwen4ExpForConditionalGeneration",
                        model_class=object,
                    ) as acknowledge:
                        runner = qwen_module.FusedMoE(prefix="layer.0")
                        tamper(runner.routed_experts)
                        acknowledge(object())
                self.assertIs(qwen_module.FusedMoE, original)

    def test_wrong_module_binding_or_unverified_device_class_fails_closed(self):
        module = _installer_module()
        installer, qwen_module, original, _calls = _fixture(module)
        qwen_module.FusedMoE = lambda **kwargs: kwargs
        plan = plan_q32(tuple(range(32)))

        with self.assertRaisesRegex(RuntimeError, "binding changed"):
            with installer.activate(
                plan=plan,
                architecture="Qwen4ExpForConditionalGeneration",
                model_class=object,
            ):
                pass
        qwen_module.FusedMoE = original

        with self.assertRaisesRegex(ValueError, "CHECKPOINT_SLICE_BINDING_UNRESOLVED"):
            module.ModelScopedSelectivePreallocationInstaller(
                qwen_module=qwen_module,
                ascend_fused_moe=original,
                ascend_api_identity=module.ASCEND_FUSED_MOE_FORWARDING_API,
                routed_experts_cls=object,
                routed_experts_api_identity="unverified",
                expected_layer_count=2,
            )

    def test_routed_only_scoped_installer_is_rejected_without_hybrid_runner(self):
        installer_module = _installer_module()
        adapter = importlib.import_module("round16.b.native_vllm_adapter")
        installer, qwen_module, original, _calls = _fixture(
            installer_module, expected_layer_count=2
        )

        class NativeModel:
            def __init__(self, *, vllm_config, prefix: str = "model") -> None:
                self.vllm_config = vllm_config
                self.prefix = prefix
                self.expert_runners = (
                    qwen_module.FusedMoE(prefix=f"{prefix}.layers.0.mlp.experts"),
                    qwen_module.FusedMoE(prefix=f"{prefix}.layers.1.mlp.experts"),
                )
                self.kv_cache = object()

            def load_weights(self, weights):
                return {name for name, _value in weights}

            def __call__(self, value):
                return self.vllm_config, value

        class Worker:
            def __init__(
                self,
                vllm_config,
                local_rank,
                rank,
                distributed_init_method,
                is_driver_worker=False,
                **kwargs,
            ) -> None:
                pass

            def init_device(self):
                pass

            def load_model(self):
                pass

        class Runner:
            def __init__(self, vllm_config, device) -> None:
                pass

            def load_model(self):
                pass

        class MoeMethod:
            def process_weights_after_loading(self, layer):
                pass

            def apply(
                self,
                layer,
                x,
                use_grouped_topk,
                top_k,
                router_logits,
                renormalize,
                routed_scaling_factor=1.0,
                expert_map=None,
                log2phy=None,
            ):
                pass

        def get_model(*, vllm_config):
            return vllm_config

        NativeModel.__module__ = "vllm_ascend.models.qwen4_exp.model"
        NativeModel.__name__ = "AscendQwen4ExpForConditionalGeneration"
        Worker.__module__ = "vllm_ascend.worker.worker"
        Worker.__name__ = "NPUWorker"
        Runner.__module__ = "vllm_ascend.worker.model_runner_v1"
        Runner.__name__ = "NPUModelRunner"
        MoeMethod.__module__ = "vllm_ascend.ops.fused_moe.fused_moe"
        MoeMethod.__name__ = "AscendUnquantizedFusedMoEMethod"
        get_model.__module__ = "vllm.model_executor.model_loader"
        get_model.__name__ = "get_model"

        registry = SimpleNamespace(calls=[])

        def register_model(architecture, model_class):
            registry.calls.append((architecture, model_class))

        registry.register_model = register_model
        receipt = adapter.register_frozen_native_plugin(
            registry,
            enabled=True,
            entrypoints=adapter.NativeEntrypoints(
                model_class=NativeModel,
                worker_class=Worker,
                model_runner_class=Runner,
                model_loader=get_model,
                moe_method_class=MoeMethod,
            ),
            installer=installer,
            resident_logical_ids=tuple(range(32)),
            hybrid_bridge_factory=lambda native: (
                lambda value: (native.vllm_config, value)
            ),
        )
        self.assertFalse(receipt.registered)
        self.assertEqual(receipt.status, "RUNTIME_RUNNER_BINDING_UNRESOLVED")
        self.assertEqual(registry.calls, [])
        self.assertIs(qwen_module.FusedMoE, original)
        self.assertFalse(installer.runtime_runner_binding_verified)

    def test_source_bound_q32_owner_and_runner_are_injected_as_one_pair(self):
        installer_module = _installer_module()
        runner_module = importlib.import_module(
            "round16.b.source_bound_hybrid_runner"
        )
        lifecycle_module = importlib.import_module("round16.b.lifecycle_bridge")
        backend_module = importlib.import_module("round16.b.kt_expert_backend")
        plan = plan_q32(tuple(range(32)))
        calls = []

        def route_once(**_kwargs):
            return runner_module.FrozenTopK((), (), valid_rows=())

        route_once.round16_api_identity = runner_module.SOURCE_BOUND_ROUTE_ONCE_API

        def native_execute(*_args):
            return {}

        native_execute.round16_api_identity = (
            runner_module.SOURCE_BOUND_NATIVE_PRESELECTED_API
        )
        dependencies = runner_module.SourceBoundRunnerDependencies(
            lifecycle=lifecycle_module.RequestLeaseManager(slot_count=1),
            route_once=route_once,
            native_preselected_execute=native_execute,
            cpu_backend=backend_module.KTExpertBackend(lambda _task: ()),
        )

        def ascend_fused_moe(*_args, **kwargs):
            calls.append(dict(kwargs))
            routed = kwargs["routed_experts_cls"](
                **dict(kwargs["routed_experts_args"])
            )
            return kwargs["runner_cls"](
                layer_name=kwargs["prefix"],
                moe_config=object(),
                router=object(),
                routed_experts=routed,
                **dict(kwargs["runner_args"]),
            )

        ascend_fused_moe.__name__ = "_ascend_FusedMoE"
        qwen_module = SimpleNamespace(
            __name__="vllm.model_executor.models.qwen3_next",
            FusedMoE=ascend_fused_moe,
        )
        installer = installer_module.ModelScopedSelectivePreallocationInstaller(
            qwen_module=qwen_module,
            ascend_fused_moe=ascend_fused_moe,
            ascend_api_identity=installer_module.ASCEND_FUSED_MOE_FORWARDING_API,
            routed_experts_cls=_VerifiedRoutedExperts,
            routed_experts_api_identity=installer_module.ROUTED_EXPERTS_Q32_API,
            hybrid_runner_cls=runner_module.SourceBoundHybridRunner,
            hybrid_runner_api_identity=(
                runner_module.SOURCE_BOUND_HYBRID_RUNNER_API
            ),
            hybrid_runner_args={"round16_runner_dependencies": dependencies},
            expected_layer_count=1,
        )

        with installer.activate(
            plan=plan,
            architecture="Qwen4ExpForConditionalGeneration",
            model_class=object,
        ) as acknowledge:
            runner = qwen_module.FusedMoE(prefix="model.layers.0.mlp.experts")
            self.assertTrue(acknowledge(object()))

        self.assertIsInstance(runner, runner_module.SourceBoundHybridRunner)
        self.assertIs(calls[0]["runner_cls"], runner_module.SourceBoundHybridRunner)
        self.assertIsInstance(calls[0]["runner_args"], MappingProxyType)
        self.assertIs(calls[0]["runner_args"]["round16_selective_plan"], plan)
        self.assertTrue(installer.source_bound_runner_pairing_verified)
        self.assertFalse(installer.runtime_runner_binding_verified)

    def test_paired_installer_rejects_any_preexisting_runner_override_atomically(self):
        installer_module = _installer_module()
        runner_module = importlib.import_module(
            "round16.b.source_bound_hybrid_runner"
        )
        plan = plan_q32(tuple(range(32)))
        calls = []

        def ascend_fused_moe(*_args, **kwargs):
            calls.append(kwargs)
            raise AssertionError("conflict must fail before original factory")

        ascend_fused_moe.__name__ = "_ascend_FusedMoE"
        qwen_module = SimpleNamespace(
            __name__="vllm.model_executor.models.qwen3_next",
            FusedMoE=ascend_fused_moe,
        )
        installer = installer_module.ModelScopedSelectivePreallocationInstaller(
            qwen_module=qwen_module,
            ascend_fused_moe=ascend_fused_moe,
            ascend_api_identity=installer_module.ASCEND_FUSED_MOE_FORWARDING_API,
            routed_experts_cls=_VerifiedRoutedExperts,
            routed_experts_api_identity=installer_module.ROUTED_EXPERTS_Q32_API,
            hybrid_runner_cls=runner_module.SourceBoundHybridRunner,
            hybrid_runner_api_identity=(
                runner_module.SOURCE_BOUND_HYBRID_RUNNER_API
            ),
            hybrid_runner_args={"round16_runner_dependencies": object()},
            expected_layer_count=1,
        )
        with installer.activate(
            plan=plan,
            architecture="Qwen4ExpForConditionalGeneration",
            model_class=object,
        ):
            with self.assertRaisesRegex(RuntimeError, "runner override"):
                qwen_module.FusedMoE(
                    prefix="model.layers.0.mlp.experts",
                    runner_cls=object,
                )
        self.assertEqual(calls, [])


if __name__ == "__main__":
    unittest.main()
