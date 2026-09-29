from __future__ import annotations

import importlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest


ROOT = Path(__file__).resolve().parents[2]
EXACT_ASCEND_ROOT = (
    ROOT
    / "artifacts"
    / "round16-b"
    / "exact-worker-source"
    / "vllm_ascend"
)


def _base_receipt(path: Path) -> None:
    source = importlib.import_module("round16.b.native_source_preflight")
    observed = {
        item.name: item.sha256 for item in source.DEFAULT_SOURCE_EXPECTATIONS
    }
    observed["model_config"] = source.DEFAULT_MODEL_CONFIG_EXPECTATION.sha256
    checks = [item.name for item in source.DEFAULT_SOURCE_EXPECTATIONS]
    checks.append("model_config")
    path.write_text(
        json.dumps(
            {
                "status": "PASS_EXACT_SOURCE_AND_API",
                "passed": True,
                "observed_sha256": observed,
                "api_checks": checks,
            }
        )
    )


def _base_token():
    source = importlib.import_module("round16.b.native_source_preflight")
    observed = tuple(
        (item.name, item.sha256) for item in source.DEFAULT_SOURCE_EXPECTATIONS
    ) + (("model_config", source.DEFAULT_MODEL_CONFIG_EXPECTATION.sha256),)
    checks = tuple(item.name for item in source.DEFAULT_SOURCE_EXPECTATIONS) + (
        "model_config",
    )
    return source.PreflightResult(
        "PASS_EXACT_SOURCE_AND_API", True, observed, checks
    ).verification_token()


def _worker_token():
    hook = importlib.import_module("round16.b.exact_donor_worker_hook")
    return hook.validate_exact_worker_source_identity(
        dict(hook.EXACT_WORKER_SOURCE_SHA256)
    )


class ExactWorkerPreflightTests(unittest.TestCase):
    def test_frozen_worker_sources_issue_receipt_consumed_by_hook(self):
        preflight = importlib.import_module(
            "round16_integration.b2_exact_worker_preflight"
        )
        hook = importlib.import_module("round16.b.exact_donor_worker_hook")
        with tempfile.TemporaryDirectory() as directory:
            receipt = Path(directory) / "base.json"
            _base_receipt(receipt)
            result = preflight.run_preflight(
                vllm_ascend_root=EXACT_ASCEND_ROOT,
                base_source_receipt=receipt,
            )
        self.assertEqual(result["status"], "PASS_B2_EXACT_WORKER_SOURCE")
        self.assertFalse(result["live_eligible"])
        self.assertFalse(result["device_executed"])
        self.assertEqual(
            result["observed_sha256"], dict(hook.EXACT_WORKER_SOURCE_SHA256)
        )
        token = hook.load_exact_worker_source_receipt(result)
        self.assertTrue(hook.validate_exact_worker_source_token(token))

    def test_api_shape_validator_rejects_missing_worker_order(self):
        preflight = importlib.import_module(
            "round16_integration.b2_exact_worker_preflight"
        )
        with self.assertRaisesRegex(ValueError, "API markers"):
            preflight.validate_worker_api_shapes(
                "worker.py", "class NPUWorker:\n    pass\n"
            )

    def test_receipt_tampering_is_rejected(self):
        preflight = importlib.import_module(
            "round16_integration.b2_exact_worker_preflight"
        )
        hook = importlib.import_module("round16.b.exact_donor_worker_hook")
        with tempfile.TemporaryDirectory() as directory:
            receipt = Path(directory) / "base.json"
            _base_receipt(receipt)
            result = preflight.run_preflight(
                vllm_ascend_root=EXACT_ASCEND_ROOT,
                base_source_receipt=receipt,
            )
        result["worker_source_binding_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "binding"):
            hook.load_exact_worker_source_receipt(result)


class ExactWorkerPluginTests(unittest.TestCase):
    def test_shipped_unresolved_factories_fail_before_device_or_model_work(self):
        unresolved = importlib.import_module(
            "round16_integration.b2_unresolved_bindings"
        )
        with self.assertRaisesRegex(
            RuntimeError, "SELECTIVE_Q32_PREALLOCATION_REQUIRED"
        ):
            unresolved.create_b2_bindings()
        with self.assertRaisesRegex(
            RuntimeError, "SELECTIVE_Q32_PREALLOCATION_REQUIRED"
        ):
            unresolved.create_p1_runtime_binding(object())

    def test_current_binding_factory_resolves_comm_and_stream_inside_forward(self):
        plugin = importlib.import_module(
            "round16_integration.b2_exact_worker_plugin"
        )
        loader = importlib.import_module("round16.b.selective_loader")
        plan = loader.plan_q32(tuple(range(32)))

        class FakeTorch:
            int32 = "int32"
            long = "long"

        for name in (
            "as_tensor",
            "clamp",
            "take",
            "ge",
            "lt",
            "logical_and",
            "arange",
            "full_like",
            "zeros_like",
            "where",
            "add",
        ):
            setattr(FakeTorch, name, staticmethod(lambda *_args, **_kwargs: None))
        tensor_ops = importlib.import_module(
            "round16_integration.p1_allgather_q32_probe"
        ).TorchNpuTensorOps(FakeTorch)

        class AllGatherCommImpl:
            def __init__(self):
                self.moe_config = SimpleNamespace(ep_size=1, tp_size=1)

            def prepare(self, **_kwargs):
                raise AssertionError("binding construction must not execute prepare")

            def fused_experts(self, **_kwargs):
                raise AssertionError("binding construction must not execute fused")

            def finalize(self, **_kwargs):
                raise AssertionError("binding construction must not execute finalize")

        first_comm = AllGatherCommImpl()
        context = SimpleNamespace(moe_comm_method=first_comm)
        stream = object()

        def get_context():
            return context

        def get_stream():
            return stream

        def build_input(**kwargs):
            return kwargs

        binding_factory = plugin.make_current_allgather_tensor_binding_factory(
            get_forward_context=get_context,
            current_stream=get_stream,
            build_fused_experts_input=build_input,
            quant_none="NONE",
            tensor_ops=tensor_ops,
        )
        owner = SimpleNamespace(
            round16_selective_plan=plan,
            w13_weight=SimpleNamespace(shape=(32, 2560, 1280), dtype="bfloat16"),
            w2_weight=SimpleNamespace(shape=(32, 640, 2560), dtype="bfloat16"),
        )
        first = binding_factory(
            layer_name="language_model.model.layers.0.mlp.experts",
            runner=object(),
            routed_owner=owner,
            plan=plan,
            source_token=_worker_token(),
            source_preflight_token=_base_token(),
        )
        self.assertIs(first.layer.comm_method, first_comm)
        self.assertIs(first.layer.comm_method_provider(), first_comm)
        self.assertIs(first.layer.stream_identity_provider(), stream)

        second_comm = AllGatherCommImpl()
        context.moe_comm_method = second_comm
        second = binding_factory(
            layer_name="language_model.model.layers.0.mlp.experts",
            runner=object(),
            routed_owner=owner,
            plan=plan,
            source_token=_worker_token(),
            source_preflight_token=_base_token(),
        )
        self.assertIs(second.layer.comm_method, second_comm)

    def test_enabled_plugin_rejects_missing_verified_q32_owner_before_patch(self):
        plugin = importlib.import_module(
            "round16_integration.b2_exact_worker_plugin"
        )
        loader = importlib.import_module("round16.b.selective_loader")

        class NPUModelRunner:
            def load_model(self):
                return None

        NPUModelRunner.__module__ = "vllm_ascend.worker.model_runner_v1"
        original = NPUModelRunner.load_model

        def ascend_fused_moe(**_kwargs):
            raise AssertionError("missing Q32 owner must fail before construction")

        ascend_fused_moe.__name__ = "_ascend_FusedMoE"

        def runtime_factory(**_kwargs):
            raise AssertionError("missing Q32 owner must fail before registration")

        runtime_factory.round16_api_identity = (
            "round16-b:exact-worker-layer-runtime-factory:v1"
        )
        bindings = plugin.B2ExactWorkerPluginBindings(
            model_runner_cls=NPUModelRunner,
            qwen_module=SimpleNamespace(
                __name__="vllm.model_executor.models.qwen3_next",
                FusedMoE=ascend_fused_moe,
            ),
            ascend_fused_moe=ascend_fused_moe,
            routed_experts_cls=object,
            upstream_moe_runner_cls=object,
            shared_order_no_overlap=object(),
            shared_order_overlap=object(),
            plan=loader.plan_q32(tuple(range(32))),
            worker_source_token=_worker_token(),
            runtime_factory=runtime_factory,
        )
        with self.assertRaisesRegex(
            RuntimeError, "SELECTIVE_Q32_PREALLOCATION_REQUIRED"
        ):
            plugin.install_b2_exact_worker_plugin(bindings, enabled=True)
        self.assertIs(NPUModelRunner.load_model, original)


if __name__ == "__main__":
    unittest.main()
