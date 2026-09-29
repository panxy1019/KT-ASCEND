from __future__ import annotations

from contextlib import contextmanager
from dataclasses import replace
import importlib
import json
from pathlib import Path
import unittest


def _adapter_module():
    try:
        return importlib.import_module("round16.b.native_vllm_adapter")
    except ModuleNotFoundError as error:
        raise AssertionError("round16.b.native_vllm_adapter must exist") from error


class _Registry:
    def __init__(self) -> None:
        self.calls: list[tuple[str, object]] = []

    def register_model(self, architecture: str, implementation: object) -> None:
        self.calls.append((architecture, implementation))


class _NativeModel:
    active_plan = None
    events: list[str] = []

    def __init__(self, *, vllm_config, prefix: str = "model") -> None:
        type(self).events.append("native_construct")
        self.vllm_config = vllm_config
        self.prefix = prefix
        self.consumed_plan = type(self).active_plan
        self.kv_cache = object()

    def load_weights(self, weights):
        return {name for name, _value in weights}

    def __call__(self, value):
        return self.vllm_config, self.prefix, value


class _Worker:
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

    def load_model(self) -> None:
        pass


class _Runner:
    def __init__(self, vllm_config, device) -> None:
        pass

    def load_model(self) -> None:
        pass


class _MoeMethod:
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
        num_experts=-1,
        expert_map=None,
        log2phy=None,
    ):
        pass


_NativeModel.__module__ = "vllm_ascend.models.qwen4_exp.model"
_NativeModel.__name__ = "AscendQwen4ExpForConditionalGeneration"
_Worker.__module__ = "vllm_ascend.worker.worker"
_Worker.__name__ = "NPUWorker"
_Runner.__module__ = "vllm_ascend.worker.model_runner_v1"
_Runner.__name__ = "NPUModelRunner"
_MoeMethod.__module__ = "vllm_ascend.ops.fused_moe.fused_moe"
_MoeMethod.__name__ = "AscendUnquantizedFusedMoEMethod"


def _get_model(*, vllm_config):
    return vllm_config


_get_model.__module__ = "vllm.model_executor.model_loader"
_get_model.__name__ = "get_model"


class _Installer:
    api_identity = "round16-b:model-scoped-selective-preallocation:v1"
    routed_experts_binding_verified = True
    runtime_runner_binding_verified = True
    hybrid_runner_api_identity = "round16-b:matched-hybrid-runner:v1"

    def __init__(self, *, consumed: bool = True, raise_on_enter: bool = False) -> None:
        self.consumed = consumed
        self.raise_on_enter = raise_on_enter
        self.events: list[object] = []

    @contextmanager
    def activate(self, *, plan, architecture, model_class):
        self.events.append(("activate", architecture, plan.resident_logical_ids))
        if self.raise_on_enter:
            raise RuntimeError("installer failed before native construction")
        model_class.active_plan = plan
        model_class.events.append("allocation_installed")
        try:
            yield lambda native: self.consumed and native.consumed_plan is plan
        finally:
            self.events.append("rollback")
            model_class.events.append("allocation_retired")
            model_class.active_plan = None


def _entrypoints(module, *, model_class=_NativeModel):
    return module.NativeEntrypoints(
        model_class=model_class,
        worker_class=_Worker,
        model_runner_class=_Runner,
        model_loader=_get_model,
        moe_method_class=_MoeMethod,
    )


def _preflight_token():
    preflight = importlib.import_module("round16.b.native_source_preflight")
    observed = tuple(
        (item.name, item.sha256) for item in preflight.DEFAULT_SOURCE_EXPECTATIONS
    ) + (("model_config", preflight.DEFAULT_MODEL_CONFIG_EXPECTATION.sha256),)
    result = preflight.PreflightResult(
        "PASS_EXACT_SOURCE_AND_API",
        True,
        observed,
        tuple(item.name for item in preflight.DEFAULT_SOURCE_EXPECTATIONS)
        + ("model_config",),
    )
    return result.verification_token()


def _runtime_capacity(module):
    return module.RuntimeContextCapacity(
        max_input_tokens=128,
        max_output_tokens=128,
        max_model_len=256,
        state_capacity=256,
    )


class NativeVllmAdapterTests(unittest.TestCase):
    def setUp(self) -> None:
        _NativeModel.events = []
        _NativeModel.active_plan = None

    def test_frozen_identity_matches_checked_in_offline_source_fixture(self) -> None:
        module = _adapter_module()
        fixture_path = (
            Path(__file__).parent / "fixtures" / "vllm_ascend_0_26_api.json"
        )
        fixture = json.loads(fixture_path.read_text(encoding="utf-8"))

        self.assertEqual(module.FROZEN_API_IDENTITY.as_dict(), fixture)
        self.assertEqual(
            module.FROZEN_API_IDENTITY.source_archive_sha256,
            "005d731a31cc11854878efbd2c2a68d1840c72afba83b048422678942773eaef",
        )
        self.assertEqual(
            module.FROZEN_API_IDENTITY.model_config_architectures,
            ("Qwen4ExpForConditionalGeneration",),
        )
        self.assertEqual(module.FROZEN_API_IDENTITY.model_config_num_hidden_layers, 48)
        self.assertEqual(module.FROZEN_API_IDENTITY.model_config_dtype, "bfloat16")
        self.assertGreaterEqual(
            module.FROZEN_API_IDENTITY.model_config_max_position_embeddings,
            256,
        )
        self.assertEqual(len(module.FROZEN_API_IDENTITY.upstream_vllm_source_sha256), 9)
        self.assertEqual(len(module.FROZEN_API_IDENTITY.source_sha256), 16)

    def test_default_off_has_no_registry_installer_or_factory_side_effect(self) -> None:
        module = _adapter_module()
        registry = _Registry()

        receipt = module.register_frozen_native_plugin(registry, enabled=False)

        self.assertEqual(receipt.status, "DISABLED_DEFAULT_OFF")
        self.assertFalse(receipt.registered)
        self.assertEqual(registry.calls, [])
        self.assertEqual(_NativeModel.events, [])

    def test_enabled_without_selective_preallocation_seam_fails_closed(self) -> None:
        module = _adapter_module()
        registry = _Registry()

        receipt = module.register_frozen_native_plugin(
            registry,
            enabled=True,
            entrypoints=_entrypoints(module),
            resident_logical_ids=tuple(range(32)),
            hybrid_bridge_factory=lambda native: lambda value: (native, value),
        )

        self.assertEqual(
            receipt.status, "CHECKPOINT_SLICE_BINDING_UNRESOLVED"
        )
        self.assertFalse(receipt.registered)
        self.assertEqual(registry.calls, [])

    def test_enabled_without_verified_device_routed_experts_fails_closed(self) -> None:
        module = _adapter_module()
        registry = _Registry()

        class UnverifiedInstaller(_Installer):
            routed_experts_binding_verified = False

        receipt = module.register_frozen_native_plugin(
            registry,
            enabled=True,
            entrypoints=_entrypoints(module),
            installer=UnverifiedInstaller(),
            resident_logical_ids=tuple(range(32)),
            hybrid_bridge_factory=lambda native: lambda value: value,
        )

        self.assertEqual(receipt.status, "CHECKPOINT_SLICE_BINDING_UNRESOLVED")
        self.assertFalse(receipt.registered)
        self.assertEqual(registry.calls, [])

    def test_q32_owner_without_matched_hybrid_runner_fails_closed(self) -> None:
        module = _adapter_module()
        registry = _Registry()

        class RoutedOnlyInstaller(_Installer):
            runtime_runner_binding_verified = False
            hybrid_runner_api_identity = None

        receipt = module.register_frozen_native_plugin(
            registry,
            enabled=True,
            entrypoints=_entrypoints(module),
            installer=RoutedOnlyInstaller(),
            resident_logical_ids=tuple(range(32)),
            hybrid_bridge_factory=lambda native: lambda value: value,
        )

        self.assertEqual(receipt.status, "RUNTIME_RUNNER_BINDING_UNRESOLVED")
        self.assertFalse(receipt.registered)
        self.assertEqual(registry.calls, [])

    def test_identity_or_entrypoint_shape_mismatch_precedes_installer_side_effect(self) -> None:
        module = _adapter_module()
        registry = _Registry()
        installer = _Installer()
        wrong_identity = replace(module.FROZEN_API_IDENTITY, vllm_version="0.26.1")

        wrong_identity_receipt = module.register_frozen_native_plugin(
            registry,
            enabled=True,
            identity=wrong_identity,
            entrypoints=_entrypoints(module),
            installer=installer,
            resident_logical_ids=tuple(range(32)),
            hybrid_bridge_factory=lambda native: lambda value: value,
        )

        class BadRunner:
            def __init__(self, config):
                pass

        BadRunner.__module__ = "vllm_ascend.worker.model_runner_v1"
        BadRunner.__name__ = "NPUModelRunner"
        bad_entrypoints = replace(
            _entrypoints(module), model_runner_class=BadRunner
        )
        bad_shape_receipt = module.register_frozen_native_plugin(
            registry,
            enabled=True,
            entrypoints=bad_entrypoints,
            installer=installer,
            resident_logical_ids=tuple(range(32)),
            hybrid_bridge_factory=lambda native: lambda value: value,
        )

        self.assertEqual(
            wrong_identity_receipt.status, "FROZEN_API_IDENTITY_MISMATCH"
        )
        self.assertEqual(bad_shape_receipt.status, "VERSION_API_SHAPE_MISMATCH")
        self.assertEqual(registry.calls, [])
        self.assertEqual(installer.events, [])

    def test_verified_path_installs_q32_before_native_constructor_and_keeps_owner(self) -> None:
        module = _adapter_module()
        registry = _Registry()
        installer = _Installer()
        resident = tuple(range(17, 49))

        receipt = module.register_frozen_native_plugin(
            registry,
            enabled=True,
            entrypoints=_entrypoints(module),
            installer=installer,
            resident_logical_ids=resident,
            source_preflight_token=_preflight_token(),
            runtime_context_capacity=_runtime_capacity(module),
            hybrid_bridge_factory=lambda native: (
                lambda value: ("hybrid", native.vllm_config, value)
            ),
        )
        registered_class = registry.calls[0][1]
        model = registered_class(vllm_config="cfg", prefix="language_model")

        self.assertEqual(receipt.status, "REGISTERED_SOURCE_BOUND_OFFLINE")
        self.assertTrue(receipt.registered)
        self.assertEqual(
            _NativeModel.events,
            ["allocation_installed", "native_construct", "allocation_retired"],
        )
        self.assertEqual(installer.events[0], ("activate", module.TARGET_ARCHITECTURE, resident))
        self.assertEqual(installer.events[-1], "rollback")
        self.assertTrue(issubclass(registered_class, _NativeModel))
        self.assertIs(model.native_model, model)
        self.assertIs(model.native_model.consumed_plan, receipt.plan)
        self.assertIs(model.kv_cache, model.native_model.kv_cache)
        self.assertEqual(model("token"), ("cfg", "language_model", "token"))
        self.assertEqual(
            model.run_hybrid_experts("hidden"), ("hybrid", "cfg", "hidden")
        )

    def test_unacknowledged_or_failed_activation_rolls_back_or_never_registers(self) -> None:
        module = _adapter_module()
        registry = _Registry()
        installer = _Installer(consumed=False)
        receipt = module.register_frozen_native_plugin(
            registry,
            enabled=True,
            entrypoints=_entrypoints(module),
            installer=installer,
            resident_logical_ids=tuple(range(32)),
            source_preflight_token=_preflight_token(),
            runtime_context_capacity=_runtime_capacity(module),
            hybrid_bridge_factory=lambda native: lambda value: value,
        )

        with self.assertRaisesRegex(RuntimeError, "did not acknowledge"):
            registry.calls[0][1](vllm_config="cfg")
        self.assertEqual(installer.events[-1], "rollback")
        self.assertIsNone(_NativeModel.active_plan)

        second_registry = _Registry()
        malformed_installer = object()
        rejected = module.register_frozen_native_plugin(
            second_registry,
            enabled=True,
            entrypoints=_entrypoints(module),
            installer=malformed_installer,
            resident_logical_ids=tuple(range(32)),
            hybrid_bridge_factory=lambda native: lambda value: value,
        )
        self.assertEqual(rejected.status, "VERSION_API_SHAPE_MISMATCH")
        self.assertEqual(second_registry.calls, [])

    def test_registration_requires_exact_preflight_token_and_runtime_context(self):
        module = _adapter_module()
        installer = _Installer()
        common = dict(
            enabled=True,
            entrypoints=_entrypoints(module),
            installer=installer,
            resident_logical_ids=tuple(range(32)),
            hybrid_bridge_factory=lambda native: lambda value: value,
        )
        registry = _Registry()
        missing = module.register_frozen_native_plugin(registry, **common)
        self.assertEqual(missing.status, "SOURCE_PREFLIGHT_REQUIRED")
        self.assertEqual(registry.calls, [])

        token = _preflight_token()
        bad = replace(
            token,
            observed_sha256=token.observed_sha256[:-1]
            + (("model_config", "0" * 64),),
        )
        rejected = module.register_frozen_native_plugin(
            registry,
            source_preflight_token=bad,
            runtime_context_capacity=_runtime_capacity(module),
            **common,
        )
        self.assertEqual(rejected.status, "SOURCE_PREFLIGHT_REJECTED")
        self.assertEqual(registry.calls, [])

        too_short = module.register_frozen_native_plugin(
            registry,
            source_preflight_token=token,
            runtime_context_capacity=module.RuntimeContextCapacity(
                128, 128, 128, 128
            ),
            **common,
        )
        self.assertEqual(
            too_short.status, "CONTEXT_CAPACITY_DEVICE_MODEL_PENDING"
        )
        self.assertEqual(registry.calls, [])

    def test_full_bank_or_invalid_q32_plan_is_rejected_before_registration(self) -> None:
        module = _adapter_module()
        for resident in (tuple(range(512)), tuple(range(31)) + (True,)):
            with self.subTest(resident_count=len(resident)):
                registry = _Registry()
                installer = _Installer()
                receipt = module.register_frozen_native_plugin(
                    registry,
                    enabled=True,
                    entrypoints=_entrypoints(module),
                    installer=installer,
                    resident_logical_ids=resident,
                    hybrid_bridge_factory=lambda native: lambda value: value,
                )
                self.assertEqual(receipt.status, "INVALID_Q32_PLAN")
                self.assertEqual(registry.calls, [])
                self.assertEqual(installer.events, [])


if __name__ == "__main__":
    unittest.main()
