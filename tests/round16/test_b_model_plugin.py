from __future__ import annotations

import importlib
from concurrent.futures import ThreadPoolExecutor
import unittest


def _plugin_module():
    try:
        return importlib.import_module("round16.b.model_plugin")
    except ModuleNotFoundError as error:
        raise AssertionError("round16.b.model_plugin must exist") from error


class _Registry:
    def __init__(self) -> None:
        self.model_calls: list[tuple[str, object]] = []
        self.platform_registry = {"npu": "original-platform"}
        self.loader_hook = "original-loader"

    def register_model(self, architecture: str, implementation: object) -> None:
        self.model_calls.append((architecture, implementation))


class _NativeModel:
    def __init__(self, model_id: str, *, revision: str) -> None:
        self.model_id = model_id
        self.revision = revision

    def __call__(self, value: str) -> tuple[str, str, str]:
        return (self.model_id, self.revision, value)


def _binding(module):
    if not hasattr(module, "VerifiedModelBinding"):
        raise AssertionError("model registration requires a verified binding contract")
    return module.VerifiedModelBinding(
        api_identity="verified-fake-vllm-runner-v1",
        native_factory=_NativeModel,
        hybrid_bridge_factory=lambda native: (
            lambda value: ("hybrid", native.model_id, value)
        ),
    )


class ModelPluginTests(unittest.TestCase):
    def test_target_is_the_frozen_round14a_conditional_model(self) -> None:
        module = _plugin_module()

        self.assertEqual(module.TARGET_ARCHITECTURE, "Qwen4ExpForConditionalGeneration")
        self.assertEqual(
            module.MODEL_IMPLEMENTATION,
            "round16.b.model_plugin:KTHybridQwen4ExpForConditionalGeneration",
        )

    def test_registers_only_target_model_once_and_is_worker_safe(self) -> None:
        module = _plugin_module()
        registry = _Registry()
        registrar = module.ModelPluginRegistrar()
        binding = _binding(module)

        with ThreadPoolExecutor(max_workers=4) as pool:
            receipts = list(
                pool.map(
                    lambda _: registrar.register(registry, binding=binding),
                    range(24),
                )
            )

        self.assertEqual(len(registry.model_calls), 1)
        self.assertEqual(registry.model_calls[0][0], module.TARGET_ARCHITECTURE)
        self.assertTrue(
            issubclass(
                registry.model_calls[0][1],
                module.KTHybridQwen4ExpForConditionalGeneration,
            )
        )
        self.assertEqual(sum(receipt.registered for receipt in receipts), 1)
        self.assertEqual(registry.platform_registry, {"npu": "original-platform"})
        self.assertEqual(registry.loader_hook, "original-loader")

    def test_rejects_non_target_architecture_without_touching_registry(self) -> None:
        module = _plugin_module()
        registry = _Registry()

        with self.assertRaisesRegex(ValueError, "target architecture"):
            module.ModelPluginRegistrar().register(
                registry,
                architecture="Qwen4ExpForCausalLM",
                binding=_binding(module),
            )

        self.assertEqual(registry.model_calls, [])

    def test_missing_native_registry_is_version_api_unresolved(self) -> None:
        module = _plugin_module()

        def missing_import(_name: str):
            raise ModuleNotFoundError("offline test has no vllm")

        decision = module.preflight_vllm_registry(import_module=missing_import)

        self.assertFalse(decision.supported)
        self.assertEqual(decision.reason_code, "VERSION_API_UNRESOLVED")
        self.assertTrue(decision.fallback_allowed)

    def test_unverified_scaffold_fails_closed_without_registry_mutation(self) -> None:
        module = _plugin_module()
        registry = _Registry()

        receipt = module.ModelPluginRegistrar().register(registry)

        self.assertFalse(receipt.registered)
        self.assertEqual(receipt.status, "VERSION_API_UNRESOLVED")
        self.assertEqual(registry.model_calls, [])

    def test_verified_fake_binding_constructs_and_forwards_calls(self) -> None:
        module = _plugin_module()
        registry = _Registry()
        receipt = module.ModelPluginRegistrar().register(
            registry,
            binding=_binding(module),
        )
        registered_class = registry.model_calls[0][1]

        model = registered_class("model-7", revision="revision-a")

        self.assertTrue(receipt.registered)
        self.assertEqual(model("token"), ("model-7", "revision-a", "token"))
        self.assertEqual(
            model.run_hybrid_experts("hidden"),
            ("hybrid", "model-7", "hidden"),
        )

    def test_registration_rejects_unverified_binding_objects(self) -> None:
        module = _plugin_module()
        registry = _Registry()

        with self.assertRaisesRegex(TypeError, "VerifiedModelBinding"):
            module.ModelPluginRegistrar().register(registry, binding=object())

        self.assertEqual(registry.model_calls, [])

    def test_verified_binding_rejects_blank_api_identity(self) -> None:
        module = _plugin_module()

        with self.assertRaisesRegex(ValueError, "non-empty"):
            module.VerifiedModelBinding(
                api_identity="   ",
                native_factory=_NativeModel,
                hybrid_bridge_factory=lambda native: lambda value: value,
            )


if __name__ == "__main__":
    unittest.main()
