from __future__ import annotations

from types import SimpleNamespace
import unittest

from round16.b.selective_loader import plan_q32
from round18.native_hybrid_live_binding import (
    NATIVE_HYBRID_LIVE_BINDING_API,
    attach_native_hybrid_quant_methods,
)
from round18.native_runner_seam import NATIVE_PARTIAL_SEAM_API


class _Delegate:
    def apply(self, **_kwargs):
        raise AssertionError("not executed by binding test")


class _Owner:
    def __init__(self, plan):
        self.round16_selective_plan = plan
        self.quant_method = _Delegate()
        self._modules = {"quant_method": self.quant_method}


class AscendMoERunner:
    def __init__(self, plan, layer):
        self.layer_name = layer
        self.routed_experts = _Owner(plan)
        self.round18_native_partial_api = NATIVE_PARTIAL_SEAM_API
        self.round18_selective_plan = plan
        self.round18_native_runner_class = type(self)


class NativeHybridLiveBindingTests(unittest.TestCase):
    def setUp(self):
        self.plan = plan_q32(tuple(range(32)))
        self.selector_module = SimpleNamespace(
            select_experts=lambda **_kwargs: (object(), object())
        )

    @staticmethod
    def callbacks(_layer):
        return (lambda **_kwargs: None,) * 3

    @staticmethod
    def identities(layer):
        return (
            lambda **kwargs: (layer, kwargs["generation"]),
            lambda _identity: True,
        )

    def test_attaches_to_q32_owner_without_replacing_native_runner(self):
        layers = tuple(
            (f"language_model.model.layers.{index}.mlp.experts", runner)
            for index in range(2)
            for runner in (AscendMoERunner(self.plan, str(index)),)
        )
        classes = tuple(type(runner) for _name, runner in layers)

        receipts = attach_native_hybrid_quant_methods(
            resolved_layers=layers,
            plan=self.plan,
            selector_module=self.selector_module,
            callbacks_for_layer=self.callbacks,
            identity_provider_for_layer=self.identities,
            expected_layer_count=2,
        )

        self.assertEqual(len(receipts), 2)
        for (_name, runner), runner_class in zip(layers, classes, strict=True):
            self.assertIs(type(runner), runner_class)
            self.assertEqual(
                runner.round18_native_hybrid_api,
                NATIVE_HYBRID_LIVE_BINDING_API,
            )
            self.assertIs(
                runner.routed_experts.quant_method.delegate.__class__, _Delegate
            )
            self.assertIs(
                runner.routed_experts._modules["quant_method"],
                runner.routed_experts.round18_registered_native_quant_method,
            )

    def test_structural_failure_commits_no_adapter(self):
        good = AscendMoERunner(self.plan, "good")
        bad = AscendMoERunner(self.plan, "bad")
        bad.round18_selective_plan = plan_q32(tuple(range(32, 64)))
        original = good.routed_experts.quant_method

        with self.assertRaisesRegex(ValueError, "different Q32 plans"):
            attach_native_hybrid_quant_methods(
                resolved_layers=(("good", good), ("bad", bad)),
                plan=self.plan,
                selector_module=self.selector_module,
                callbacks_for_layer=self.callbacks,
                identity_provider_for_layer=self.identities,
                expected_layer_count=2,
            )

        self.assertIs(good.routed_experts.quant_method, original)


if __name__ == "__main__":
    unittest.main()
