from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
import unittest

from round18.native_hybrid_quant_method import (
    NativeHybridQuantMethodAdapter,
    StaleHybridWorkError,
)


@dataclass
class _Result:
    routed_out: int
    marker: str = "native-result"


@dataclass(frozen=True)
class _CpuPartial:
    output: int
    route_weights_applied: bool = True
    routed_scaling_applied: bool = False


class _Delegate:
    def __init__(self, selector_module, *, fail: bool = False, calls: int = 1):
        self.selector_module = selector_module
        self.fail = fail
        self.calls = calls

    def apply(self, *, x, router_logits, **_kwargs):
        for _ in range(self.calls):
            self.selector_module.select_experts(
                hidden_states=x, router_logits=router_logits
            )
        if self.fail:
            raise RuntimeError("native branch failed")
        return _Result(10)


class NativeHybridQuantMethodTests(unittest.TestCase):
    def _fixture(
        self,
        *,
        delegate_fail=False,
        selector_calls=1,
        current=True,
        route_observer=None,
    ):
        events = []
        ids = object()
        weights = object()

        def selector(**_kwargs):
            events.append("selector")
            return weights, ids

        selector_module = SimpleNamespace(select_experts=selector)
        original_selector = selector_module.select_experts

        def identity_provider(**kwargs):
            return (
                "request-7",
                11,
                kwargs["layer_name"],
                kwargs["generation"],
            )

        def submit(**kwargs):
            events.append("submit")
            self.assertIs(kwargs["topk_ids"], ids)
            self.assertIs(kwargs["topk_weights"], weights)
            return {"identity": kwargs["identity_key"]}

        def sync(**kwargs):
            events.append("sync")
            self.assertEqual(kwargs["pending"]["identity"], kwargs["identity_key"])
            self.assertEqual(kwargs["reference_tensor"], 10)
            return _CpuPartial(3)

        def abort(**kwargs):
            events.append("abort")
            self.assertEqual(kwargs["pending"]["identity"], kwargs["identity_key"])

        def identity_is_current(identity_key):
            events.append("identity-check")
            return current

        delegate = _Delegate(
            selector_module,
            fail=delegate_fail,
            calls=selector_calls,
        )
        adapter = NativeHybridQuantMethodAdapter(
            delegate=delegate,
            selector_module=selector_module,
            layer_name="model.layers.4.mlp.experts",
            identity_provider=identity_provider,
            cpu_submit=submit,
            cpu_sync=sync,
            cpu_abort=abort,
            identity_is_current=identity_is_current,
            route_observer=route_observer,
        )
        return adapter, selector_module, original_selector, events

    def test_original_selector_runs_once_and_only_routed_partial_is_merged(self):
        adapter, selector_module, selector, events = self._fixture()

        result = adapter.apply(x="hidden", router_logits="router")

        self.assertEqual(result.routed_out, 13)
        self.assertEqual(result.marker, "native-result")
        self.assertEqual(events, ["selector", "submit", "sync", "identity-check"])
        self.assertIs(selector_module.select_experts, selector)
        self.assertEqual(adapter.completed_generations, 1)

    def test_route_observer_receives_original_selector_tensors_before_submit(self):
        observed = []

        def observer(**kwargs):
            observed.append(kwargs)

        adapter, _module, _selector, events = self._fixture(route_observer=observer)
        adapter.apply(x="hidden", router_logits="router")

        self.assertEqual(events, ["selector", "submit", "sync", "identity-check"])
        self.assertEqual(len(observed), 1)
        self.assertEqual(observed[0]["layer_name"], "model.layers.4.mlp.experts")
        self.assertEqual(observed[0]["generation"], 1)
        self.assertEqual(observed[0]["hidden_states"], "hidden")
        self.assertIsNotNone(observed[0]["topk_ids"])
        self.assertIsNotNone(observed[0]["topk_weights"])

    def test_native_failure_drains_pending_cpu_and_restores_selector(self):
        adapter, selector_module, selector, events = self._fixture(delegate_fail=True)

        with self.assertRaisesRegex(RuntimeError, "native branch failed"):
            adapter.apply(x="hidden", router_logits="router")

        self.assertEqual(events, ["selector", "submit", "abort"])
        self.assertIs(selector_module.select_experts, selector)

    def test_delegate_must_use_the_exact_selector_once(self):
        adapter, selector_module, selector, events = self._fixture(selector_calls=2)

        with self.assertRaisesRegex(RuntimeError, "exactly once"):
            adapter.apply(x="hidden", router_logits="router")

        self.assertEqual(events, ["selector", "submit", "abort"])
        self.assertIs(selector_module.select_experts, selector)

    def test_stale_epoch_is_discarded_after_drain_without_visible_merge(self):
        adapter, selector_module, selector, events = self._fixture(current=False)

        with self.assertRaises(StaleHybridWorkError):
            adapter.apply(x="hidden", router_logits="router")

        self.assertEqual(events, ["selector", "submit", "sync", "identity-check"])
        self.assertIs(selector_module.select_experts, selector)

    def test_cpu_partial_must_already_contain_topk_weight_and_no_scaling(self):
        adapter, _module, _selector, _events = self._fixture()

        def bad_sync(**_kwargs):
            return _CpuPartial(
                3, route_weights_applied=False, routed_scaling_applied=False
            )

        adapter.cpu_sync = bad_sync
        with self.assertRaisesRegex(ValueError, "original Top-K weights"):
            adapter.apply(x="hidden", router_logits="router")


if __name__ == "__main__":
    unittest.main()
