from __future__ import annotations

import importlib
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest


def _probe():
    try:
        return importlib.import_module(
            "round16_integration.p1_allgather_q32_probe"
        )
    except ModuleNotFoundError as error:
        raise AssertionError(
            "the documented p1_allgather_q32_probe module must exist"
        ) from error


class _Device:
    type = "npu"

    def __init__(self, name="npu:0"):
        self.name = name

    def __str__(self):
        return self.name


class _OpaqueTensor:
    def __init__(self, data, dtype, *, device="npu:0", name="tensor"):
        self._data = data
        self.shape = self._shape(data)
        self.dtype = dtype
        self.device = _Device(device)
        self.name = name
        self.iteration_attempts = 0
        self.scalar_attempts = 0

    @staticmethod
    def _shape(data):
        if not isinstance(data, tuple):
            return ()
        if not data:
            return (0,)
        if isinstance(data[0], tuple):
            return (len(data), len(data[0]))
        return (len(data),)

    def __iter__(self):
        self.iteration_attempts += 1
        raise AssertionError("device tensor must not be iterated")

    def item(self):
        self.scalar_attempts += 1
        raise AssertionError("device tensor scalar extraction is forbidden")

    def tolist(self):
        self.scalar_attempts += 1
        raise AssertionError("device tensor host materialization is forbidden")

    def __bool__(self):
        self.scalar_attempts += 1
        raise AssertionError("device tensor truth extraction is forbidden")


def _map_unary(value, function):
    data = value._data
    if data and isinstance(data[0], tuple):
        converted = tuple(tuple(function(item) for item in row) for row in data)
    else:
        converted = tuple(function(item) for item in data)
    return _OpaqueTensor(converted, value.dtype, device=str(value.device))


def _map_binary(left, right, function, dtype="bool"):
    left_data = left._data
    right_data = right._data if isinstance(right, _OpaqueTensor) else right
    if left_data and isinstance(left_data[0], tuple):
        rows = []
        for row_index, row in enumerate(left_data):
            converted = []
            for column_index, item in enumerate(row):
                if isinstance(right_data, tuple):
                    if right_data and isinstance(right_data[0], tuple):
                        right_row = right_data[row_index]
                        other = right_row[0] if len(right_row) == 1 else right_row[column_index]
                    else:
                        other = right_data[column_index]
                else:
                    other = right_data
                converted.append(function(item, other))
            rows.append(tuple(converted))
        data = tuple(rows)
    else:
        data = tuple(function(item, right_data) for item in left_data)
    return _OpaqueTensor(data, dtype, device=str(left.device))


class _SpyTorch:
    int32 = "int32"
    long = "int64"
    bfloat16 = "bfloat16"

    def __init__(self):
        self.calls = []

    def as_tensor(self, values, *, dtype, device):
        self.calls.append("as_tensor")
        return _OpaqueTensor(tuple(values), dtype, device=str(device), name="mapping")

    def clamp(self, value, minimum, maximum):
        self.calls.append("clamp")
        return _map_unary(value, lambda item: min(max(item, minimum), maximum))

    def take(self, mapping, indices):
        self.calls.append("take")
        return _map_unary(indices, lambda item: mapping._data[item])

    def ge(self, value, threshold):
        self.calls.append("ge")
        return _map_unary(value, lambda item: item >= threshold)

    def lt(self, value, threshold):
        self.calls.append("lt")
        return _map_unary(value, lambda item: item < threshold)

    def logical_and(self, left, right):
        self.calls.append("logical_and")
        return _map_binary(left, right, lambda a, b: bool(a and b))

    def arange(self, stop, *, device):
        self.calls.append("arange")
        return _OpaqueTensor(tuple(range(stop)), "int64", device=str(device))

    def reshape(self, value, shape):
        self.calls.append("reshape")
        rows, columns = shape
        assert columns == 1
        return _OpaqueTensor(
            tuple((item,) for item in value._data), value.dtype,
            device=str(value.device),
        )

    def full_like(self, value, fill_value):
        self.calls.append("full_like")
        return _map_unary(value, lambda _item: fill_value)

    def zeros_like(self, value):
        self.calls.append("zeros_like")
        return _map_unary(value, lambda _item: 0.0)

    def where(self, condition, left, right):
        self.calls.append("where")
        rows = []
        for mask_row, left_row, right_row in zip(
            condition._data, left._data, right._data
        ):
            rows.append(
                tuple(a if keep else b for keep, a, b in zip(mask_row, left_row, right_row))
            )
        return _OpaqueTensor(
            tuple(rows), left.dtype, device=str(left.device), name="where"
        )

    def cast(self, value, dtype):
        self.calls.append("cast")
        return _OpaqueTensor(value._data, dtype, device=str(value.device))

    def add(self, left, right):
        self.calls.append("add")
        return _map_binary(left, right, lambda a, b: a + b, dtype=left.dtype)


class P1IntegrationProbeTests(unittest.TestCase):
    def test_device_payload_maps_resident_and_zeroes_cpu_padding_without_host_read(self):
        probe = _probe()
        loader = importlib.import_module("round16.b.selective_loader")
        plan = loader.plan_q32(tuple(range(32)))
        torch = _SpyTorch()
        ops = probe.TorchNpuTensorOps(torch)
        builder = probe.DevicePhysicalTopKPayloadBuilder(plan, ops)
        ids = _OpaqueTensor(
            (
                (0, 32, -1, 31, 511, 1, 33, 2, 34, 3),
                (0, 1, 2, 3, 4, 5, 6, 7, 8, 9),
            ),
            "int32",
            name="logical-ids",
        )
        weights = _OpaqueTensor(
            ((0.1,) * 10, (0.2,) * 10), "bfloat16", name="weights"
        )
        payload = builder(
            topk_ids=ids,
            topk_weights=weights,
            resident_logical_ids=plan.resident_logical_ids,
            logical_to_physical=plan.logical_to_physical,
            valid_rows=(True, False),
            safe_physical_id=0,
            identity_key=("request", 1),
        )
        self.assertEqual(
            payload.topk_ids._data[0],
            (0, 0, 0, 31, 0, 1, 0, 2, 0, 3),
        )
        self.assertEqual(
            payload.topk_weights._data[0],
            (0.1, 0.0, 0.0, 0.1, 0.0, 0.1, 0.0, 0.1, 0.0, 0.1),
        )
        self.assertEqual(payload.topk_ids._data[1], (0,) * 10)
        self.assertEqual(payload.topk_weights._data[1], (0.0,) * 10)
        self.assertIn("take", torch.calls)
        self.assertIn("where", torch.calls)
        self.assertNotIn("synchronize", torch.calls)
        for tensor in (ids, weights, payload.topk_ids, payload.topk_weights):
            self.assertEqual(tensor.iteration_attempts, 0)
            self.assertEqual(tensor.scalar_attempts, 0)

    def test_device_payload_preserves_frozen_s0_float32_route_weights(self):
        probe = _probe()
        loader = importlib.import_module("round16.b.selective_loader")
        plan = loader.plan_q32(tuple(range(32)))
        torch = _SpyTorch()
        builder = probe.DevicePhysicalTopKPayloadBuilder(
            plan, probe.TorchNpuTensorOps(torch)
        )
        ids = _OpaqueTensor((tuple(range(10)),), "int32")
        weights = _OpaqueTensor(((0.1,) * 10,), "float32")
        payload = builder(
            topk_ids=ids,
            topk_weights=weights,
            resident_logical_ids=plan.resident_logical_ids,
            logical_to_physical=plan.logical_to_physical,
            valid_rows=(True,),
            safe_physical_id=0,
            identity_key=("float32-route", 1),
        )
        self.assertEqual(payload.topk_weights.dtype, "float32")

    def test_tensor_weighted_merge_preserves_device_tensor_and_markers(self):
        probe = _probe()
        adapter = importlib.import_module(
            "round16.b.allgather_q32_device_adapter"
        )
        torch = _SpyTorch()
        merger = probe.TensorWeightedAggregateMerger(
            probe.TorchNpuTensorOps(torch)
        )
        native_tensor = _OpaqueTensor(
            ((1.0, 2.0),), "bfloat16", name="native"
        )
        kt_tensor = _OpaqueTensor(
            ((3.0, 4.0),), "bfloat16", name="kt"
        )
        result = merger.merge(
            adapter.DeviceWeightedPartial(native_tensor),
            probe.KTDeviceWeightedPartial(kt_tensor),
        )
        self.assertEqual(result.output._data, ((4.0, 6.0),))
        self.assertTrue(result.route_weights_applied)
        self.assertFalse(result.routed_scaling_applied)
        self.assertEqual(torch.calls, ["add"])
        for tensor in (native_tensor, kt_tensor, result.output):
            self.assertEqual(tensor.iteration_attempts, 0)
            self.assertEqual(tensor.scalar_attempts, 0)

    def test_exact_snapshot_and_source_token_validate_before_runtime_import(self):
        probe = _probe()
        preflight = importlib.import_module("round16.b.native_source_preflight")
        repository = Path(__file__).resolve().parents[2]
        manifest = repository / "artifacts/round16-b/device-overlay/P1_REMOTE_SNAPSHOT_PREFLIGHT_MANIFEST.json"
        self.assertTrue(manifest.is_file())
        snapshot = probe.load_snapshot_manifest(manifest)
        self.assertEqual(snapshot.layer_count, 48)
        self.assertEqual(snapshot.resident_logical_ids, tuple(range(32)))
        self.assertEqual(snapshot.w13_shape, (32, 2560, 1280))
        self.assertEqual(snapshot.w2_shape, (32, 640, 2560))
        self.assertEqual(
            snapshot.q32_receipt_path.name, "S0_V295_Q32_PHYSICAL.json"
        )

        observed = {
            item.name: item.sha256 for item in preflight.DEFAULT_SOURCE_EXPECTATIONS
        }
        observed["model_config"] = preflight.DEFAULT_MODEL_CONFIG_EXPECTATION.sha256
        token_payload = {
            "api_checks": [
                item.name for item in preflight.DEFAULT_SOURCE_EXPECTATIONS
            ] + ["model_config"],
            "observed_sha256": observed,
            "passed": True,
            "status": "PASS_EXACT_SOURCE_AND_API",
        }
        with tempfile.TemporaryDirectory() as directory:
            token_path = Path(directory) / "token.json"
            output_path = Path(directory) / "result.json"
            token_path.write_text(json.dumps(token_payload))
            token = probe.load_source_preflight_token(token_path)
            self.assertTrue(preflight.validate_exact_source_preflight_token(token))
            result = probe.main(
                [
                    "--validate-only",
                    "--source-token", str(token_path),
                    "--snapshot-manifest", str(manifest),
                    "--output", str(output_path),
                ]
            )
            self.assertEqual(result, 0)
            receipt = json.loads(output_path.read_text())
            self.assertEqual(
                receipt["status"], "READY_FOR_DEVICE_RUNTIME_BINDING"
            )
            self.assertFalse(receipt["device_executed"])

    def test_snapshot_tamper_and_unenabled_device_run_fail_before_factory_import(self):
        probe = _probe()
        repository = Path(__file__).resolve().parents[2]
        source = repository / "artifacts/round16-b/device-overlay/P1_REMOTE_SNAPSHOT_PREFLIGHT_MANIFEST.json"
        payload = json.loads(source.read_text())
        payload["s0_evidence"]["q32_physical_sha256"] = "0" * 64
        with tempfile.TemporaryDirectory() as directory:
            bad = Path(directory) / "bad.json"
            bad.write_text(json.dumps(payload))
            with self.assertRaisesRegex(ValueError, "S0 Q32 receipt identity"):
                probe.load_snapshot_manifest(bad)

        with self.assertRaisesRegex(RuntimeError, "ROUND16_B_ENABLE_P1"):
            probe.require_device_enable({})

    def test_real_probe_orchestrator_orders_submit_native_sync_and_tensor_merge(self):
        probe = _probe()
        adapter = importlib.import_module(
            "round16.b.allgather_q32_device_adapter"
        )
        loader = importlib.import_module("round16.b.selective_loader")
        preflight = importlib.import_module("round16.b.native_source_preflight")
        runner = importlib.import_module("round16.b.source_bound_hybrid_runner")
        repository = Path(__file__).resolve().parents[2]
        snapshot = probe.load_snapshot_manifest(
            repository / "artifacts/round16-b/device-overlay/P1_REMOTE_SNAPSHOT_PREFLIGHT_MANIFEST.json"
        )
        observed = tuple(
            (item.name, item.sha256)
            for item in preflight.DEFAULT_SOURCE_EXPECTATIONS
        ) + (("model_config", preflight.DEFAULT_MODEL_CONFIG_EXPECTATION.sha256),)
        checks = tuple(
            item.name for item in preflight.DEFAULT_SOURCE_EXPECTATIONS
        ) + ("model_config",)
        token = preflight.PreflightResult(
            "PASS_EXACT_SOURCE_AND_API", True, observed, checks
        ).verification_token()
        plan = loader.SelectiveLoadPlan(
            snapshot.resident_logical_ids,
            snapshot.logical_to_physical,
        )
        events = []

        class AllGatherCommImpl:
            moe_config = SimpleNamespace(tp_size=1, ep_size=1)

            def prepare(self, **kwargs):
                events.append("prepare")
                return SimpleNamespace(
                    hidden_states=kwargs["hidden_states"],
                    router_logits=kwargs["router_logits"],
                    mc2_mask=None,
                    padded_hidden_states_shape=None,
                    pertoken_scale=None,
                )

            def fused_experts(self, *, fused_experts_input):
                events.append("native")
                self.input = fused_experts_input
                return SimpleNamespace(
                    routed_out=_OpaqueTensor(
                        ((1.0, 2.0),), "bfloat16", name="native"
                    )
                )

            def finalize(self, **kwargs):
                events.append("finalize")
                return kwargs["hidden_states"]

        comm = AllGatherCommImpl()
        stream_identity = object()

        def comm_provider():
            return comm

        comm_provider.round16_api_identity = runner.SOURCE_BOUND_MOE_COMM_CONTEXT_API

        def stream_provider():
            return stream_identity

        stream_provider.round16_api_identity = adapter.ALLGATHER_STREAM_IDENTITY_API

        def fused_builder(**kwargs):
            events.append("build")
            return SimpleNamespace(**kwargs)

        fused_builder.round16_api_identity = adapter.ALLGATHER_BUILD_FUSED_INPUT_API
        manager = SimpleNamespace(
            global_num_experts=512,
            local_num_experts=32,
            map_global_to_local=lambda logical_id: plan.logical_to_physical[logical_id],
        )
        owner = SimpleNamespace(
            round16_selective_plan=plan,
            expert_map_manager=manager,
            w13_weight=SimpleNamespace(
                shape=snapshot.w13_shape,
                dtype="bfloat16",
                device=_Device(),
            ),
            w2_weight=SimpleNamespace(
                shape=snapshot.w2_shape,
                dtype="bfloat16",
                device=_Device(),
            ),
        )
        layer_binding = adapter.AllGatherQ32LayerBinding(
            "model.layers.0.mlp.experts",
            plan,
            owner,
            comm,
            comm_provider,
            stream_provider,
            fused_builder,
            token,
        )
        ids = _OpaqueTensor((tuple(range(10)),), "int32", name="ids")
        weights = _OpaqueTensor(((0.1,) * 10,), "bfloat16", name="weights")

        def selector_once(**kwargs):
            events.append("selector")
            return adapter.DeviceSelectedTopK(ids, weights, kwargs["valid_rows"])

        selector_once.round16_api_identity = probe.P1_SELECTOR_API

        def kt_submit(**_kwargs):
            events.append("kt_submit")
            return object()

        kt_submit.round16_api_identity = probe.P1_KT_SUBMIT_API

        def kt_sync(**_kwargs):
            events.append("kt_sync")
            return probe.KTDeviceWeightedPartial(
                _OpaqueTensor(((3.0, 4.0),), "bfloat16", name="kt")
            )

        kt_sync.round16_api_identity = probe.P1_KT_SYNC_API

        def kt_abort(**_kwargs):
            events.append("kt_abort")

        kt_abort.round16_api_identity = probe.P1_KT_ABORT_API
        binding = probe.P1RuntimeBinding(
            layer_binding,
            probe.TorchNpuTensorOps(_SpyTorch()),
            _OpaqueTensor(((5.0, 6.0),), "bfloat16", name="hidden"),
            _OpaqueTensor(((0.0,) * 512,), "float32", name="router"),
            selector_once,
            kt_submit,
            kt_sync,
            kt_abort,
        )
        manager.map_global_to_local = lambda logical_id: (
            1 if logical_id == 0 else plan.logical_to_physical[logical_id]
        )
        with self.assertRaisesRegex(ValueError, "manager mapping"):
            probe.validate_live_owner(layer_binding, snapshot, 0)
        self.assertEqual(events, [])
        manager.map_global_to_local = (
            lambda logical_id: plan.logical_to_physical[logical_id]
        )
        result = probe.run_device_probe(
            binding, probe.P1FactoryContext(token, snapshot, 0, 1)
        )
        self.assertEqual(
            events,
            ["prepare", "selector", "kt_submit", "build", "native", "finalize", "kt_sync"],
        )
        self.assertEqual(result["status"], "PASS_P1_ONE_LAYER_DEVICE_CONTRACT")
        self.assertEqual(result["output_shape"], [1, 2])
        self.assertTrue(result["route_weights_applied"])
        self.assertFalse(result["routed_scaling_applied"])
        self.assertNotIn("kt_abort", events)

    def test_packaged_probe_module_has_no_torch_or_vllm_import_side_effect(self):
        probe = _probe()
        self.assertEqual(
            probe.P1_PROBE_API,
            "round16-integration:p1-allgather-q32-probe:v1",
        )
        repository = Path(__file__).resolve().parents[2]
        self.assertTrue(
            (
                repository
                / "artifacts/round16-b/device-overlay/run_p1_allgather_q32_probe.py"
            ).is_file()
        )
