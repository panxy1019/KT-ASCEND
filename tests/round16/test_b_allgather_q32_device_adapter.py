from __future__ import annotations

import hashlib
import importlib
import json
from types import SimpleNamespace
import unittest


def _modules():
    try:
        adapter = importlib.import_module(
            "round16.b.allgather_q32_device_adapter"
        )
    except ModuleNotFoundError as error:
        raise AssertionError("ALLGATHER Q32 device adapter module must exist") from error
    return (
        adapter,
        importlib.import_module("round16.b.native_source_preflight"),
        importlib.import_module("round16.b.selective_loader"),
        importlib.import_module("round16.b.source_bound_hybrid_runner"),
        importlib.import_module("round16.b.routing_bridge"),
    )


def _source_token(preflight):
    observed = tuple(
        (item.name, item.sha256) for item in preflight.DEFAULT_SOURCE_EXPECTATIONS
    ) + (("model_config", preflight.DEFAULT_MODEL_CONFIG_EXPECTATION.sha256),)
    checks = tuple(
        item.name for item in preflight.DEFAULT_SOURCE_EXPECTATIONS
    ) + ("model_config",)
    payload = json.dumps(
        {"observed_sha256": list(observed), "api_checks": list(checks)},
        separators=(",", ":"),
        sort_keys=True,
    ).encode()
    return preflight.SourcePreflightToken(
        observed,
        checks,
        hashlib.sha256(payload).hexdigest(),
    )


class _Weight:
    def __init__(self, shape):
        self.shape = shape
        self.dtype = "bfloat16"


class _TupleAllGatherCommImpl:
    def __init__(self, events, *, pad=False, fail_at=None):
        self.events = events
        self.pad = pad
        self.fail_at = fail_at
        self.moe_config = SimpleNamespace(ep_size=1, tp_size=1)
        self.last_fused_input = None

    def prepare(self, **kwargs):
        self.events.append("prepare")
        if self.fail_at == "prepare":
            raise RuntimeError("prepare failed")
        hidden = tuple(kwargs["hidden_states"])
        logits = tuple(kwargs["router_logits"])
        if self.pad:
            hidden += ((0.0, 0.0),)
            logits += ((0.0,) * 512,)
        return SimpleNamespace(
            hidden_states=hidden,
            router_logits=logits,
            mc2_mask=None,
            padded_hidden_states_shape=(len(hidden), 2) if self.pad else None,
            pertoken_scale=None,
        )

    def fused_experts(self, *, fused_experts_input):
        self.events.append("fused_experts")
        if self.fail_at == "fused_experts":
            raise RuntimeError("fused failed")
        self.last_fused_input = fused_experts_input
        rows = tuple(
            (
                sum(float(weight) for weight in weights),
                sum(float(expert_id) * float(weight) for expert_id, weight in zip(ids, weights)),
            )
            for ids, weights in zip(
                fused_experts_input.topk_ids,
                fused_experts_input.topk_weights,
            )
        )
        return SimpleNamespace(routed_out=rows)

    def finalize(self, **kwargs):
        self.events.append("finalize")
        if self.fail_at == "finalize":
            raise RuntimeError("finalize failed")
        return kwargs["hidden_states"]


class AllGatherQ32DeviceAdapterTests(unittest.TestCase):
    def _fixture(self, *, pad=False, fail_at=None):
        adapter, preflight, loader, runner, routing = _modules()
        events = []
        comm = type("AllGatherCommImpl", (_TupleAllGatherCommImpl,), {})(
            events, pad=pad, fail_at=fail_at
        )
        current = {"comm": comm, "stream": object()}
        plan = loader.plan_q32(tuple(range(32)))
        owner = SimpleNamespace(
            round16_selective_plan=plan,
            w13_weight=_Weight((32, 8, 16)),
            w2_weight=_Weight((32, 16, 8)),
            w13_bias=None,
            w2_bias=None,
            swiglu_limit=0.0,
            swiglu_alpha=1.0,
            swiglu_beta=0.0,
        )

        def current_comm():
            return current["comm"]

        current_comm.round16_api_identity = runner.SOURCE_BOUND_MOE_COMM_CONTEXT_API

        def current_stream():
            return current["stream"]

        current_stream.round16_api_identity = adapter.ALLGATHER_STREAM_IDENTITY_API

        def builder(**kwargs):
            events.append("build")
            if fail_at == "build":
                raise RuntimeError("build failed")
            return SimpleNamespace(**kwargs)

        builder.round16_api_identity = adapter.ALLGATHER_BUILD_FUSED_INPUT_API
        binding = adapter.AllGatherQ32LayerBinding(
            layer_name="model.layers.0.mlp.experts",
            plan=plan,
            routed_owner=owner,
            comm_method=comm,
            comm_method_provider=current_comm,
            stream_identity_provider=current_stream,
            build_fused_experts_input=builder,
            source_preflight_token=_source_token(preflight),
        )
        instance = adapter.AllGatherQ32OfflineFakeAdapter(binding)
        return adapter, runner, routing, plan, events, current, comm, instance

    @staticmethod
    def _routes(runner, routing, plan, rows, valid_rows=None):
        weights = tuple(tuple(0.1 for _ in row) for row in rows)
        frozen = runner.FrozenTopK(
            tuple(tuple(row) for row in rows),
            weights,
            tuple(True for _ in rows) if valid_rows is None else valid_rows,
        )
        return routing.plan_routes(
            frozen.as_route_batch(),
            resident_logical_ids=plan.resident_logical_ids,
        )

    def test_all_cpu_all_npu_mixed_and_padding_build_physical_payload(self):
        cases = (
            ("all_cpu", (tuple(range(32, 42)),), ((0,) * 10,), ((0.0,) * 10,)),
            ("all_npu", (tuple(range(10)),), (tuple(range(10)),), ((0.1,) * 10,)),
            (
                "mixed",
                ((0, 32, 1, 33, 2, 34, 3, 35, 4, 36),),
                ((0, 0, 1, 0, 2, 0, 3, 0, 4, 0),),
                ((0.1, 0.0, 0.1, 0.0, 0.1, 0.0, 0.1, 0.0, 0.1, 0.0),),
            ),
        )
        for name, logical_rows, expected_ids, expected_weights in cases:
            with self.subTest(name=name):
                _adapter, _runner, routing, plan, events, _current, comm, instance = self._fixture()
                prepared = instance.prepare(((5.0, 6.0),), ((0.0,) * 512,), "model.layers.0.mlp.experts")
                route_plan = self._routes(_runner, routing, plan, logical_rows)
                result = instance.execute(
                    prepared,
                    route_plan,
                    prepared.selector_hidden_states,
                    ("work", name),
                )
                self.assertEqual(comm.last_fused_input.topk_ids, expected_ids)
                self.assertEqual(comm.last_fused_input.topk_weights, expected_weights)
                self.assertIsNone(comm.last_fused_input.expert_map)
                self.assertIsNone(comm.last_fused_input.log2phy)
                self.assertEqual(comm.last_fused_input.quant_type, "NONE")
                self.assertTrue(result.route_weights_applied)
                self.assertFalse(result.routed_scaling_applied)
                self.assertEqual(events, ["prepare", "build", "fused_experts", "finalize"])

        _adapter, runner, routing, plan, events, _current, comm, instance = self._fixture(pad=True)
        prepared = instance.prepare(((5.0, 6.0),), ((0.0,) * 512,), "model.layers.0.mlp.experts")
        self.assertEqual(prepared.valid_rows, (True, False))
        routes = self._routes(
            runner,
            routing,
            plan,
            ((0, 32, 1, 33, 2, 34, 3, 35, 4, 36), (40,) * 10),
            (True, False),
        )
        result = instance.execute(prepared, routes, prepared.selector_hidden_states, ("work", "pad"))
        self.assertEqual(comm.last_fused_input.topk_ids[1], (0,) * 10)
        self.assertEqual(comm.last_fused_input.topk_weights[1], (0.0,) * 10)
        self.assertEqual(result.output[1], (0.0, 0.0))

    def test_layout_comm_and_stream_identity_fail_closed(self):
        adapter, runner, routing, plan, events, current, _comm, instance = self._fixture()
        prepared = instance.prepare(((5.0, 6.0),), ((0.0,) * 512,), "model.layers.0.mlp.experts")
        routes = self._routes(runner, routing, plan, (tuple(range(10)),))
        current["stream"] = object()
        with self.assertRaisesRegex(RuntimeError, "stream identity"):
            instance.execute(
                prepared,
                routes,
                prepared.selector_hidden_states,
                ("work", "stream"),
            )
        self.assertEqual(events, ["prepare"])

        _a, runner, routing, plan, _events, current, _c, second = self._fixture()
        prepared = second.prepare(((5.0, 6.0),), ((0.0,) * 512,), "model.layers.0.mlp.experts")
        routes = self._routes(runner, routing, plan, (tuple(range(10)),))
        current["comm"] = object()
        with self.assertRaisesRegex(RuntimeError, "comm object identity"):
            second.execute(
                prepared,
                routes,
                prepared.selector_hidden_states,
                ("work", "comm"),
            )

        with self.assertRaisesRegex(ValueError, "TP1"):
            adapter.AllGatherRuntimeContract(tp_size=2)

    def test_builder_fused_and_finalize_failures_clear_one_shot_state(self):
        for fail_at in ("build", "fused_experts", "finalize"):
            with self.subTest(fail_at=fail_at):
                _adapter, runner, routing, plan, _events, _current, comm, instance = self._fixture(fail_at=fail_at)
                prepared = instance.prepare(((5.0, 6.0),), ((0.0,) * 512,), "model.layers.0.mlp.experts")
                routes = self._routes(runner, routing, plan, (tuple(range(10)),))
                with self.assertRaisesRegex(RuntimeError, "failed"):
                    instance.execute(prepared, routes, prepared.selector_hidden_states, ("work", fail_at))
                comm.fail_at = None
                next_prepared = instance.prepare(((7.0, 8.0),), ((0.0,) * 512,), "model.layers.0.mlp.experts")
                instance.cleanup(next_prepared)

    def test_callback_identities_bind_directly_to_runner_dependencies(self):
        adapter, runner, _routing, _plan, _events, _current, _comm, instance = self._fixture()
        self.assertEqual(
            getattr(instance.prepare, "round16_api_identity"),
            runner.SOURCE_BOUND_MOE_COMM_PREPARE_API,
        )
        self.assertEqual(
            getattr(instance.execute, "round16_api_identity"),
            runner.SOURCE_BOUND_NATIVE_WEIGHTED_PARTIAL_API,
        )
        self.assertEqual(
            getattr(instance.cleanup, "round16_api_identity"),
            runner.SOURCE_BOUND_MOE_COMM_CLEANUP_API,
        )

    def test_unverified_source_or_non_bfloat16_q32_layout_allocates_nothing(self):
        adapter, preflight, loader, runner, _routing = _modules()
        plan = loader.plan_q32(tuple(range(32)))
        events = []
        comm = type("AllGatherCommImpl", (_TupleAllGatherCommImpl,), {})(events)
        owner = SimpleNamespace(
            round16_selective_plan=plan,
            w13_weight=SimpleNamespace(shape=(32, 8, 16), dtype="float16"),
            w2_weight=_Weight((32, 16, 8)),
        )
        provider = lambda: comm
        provider.round16_api_identity = runner.SOURCE_BOUND_MOE_COMM_CONTEXT_API
        stream = lambda: "stream"
        stream.round16_api_identity = adapter.ALLGATHER_STREAM_IDENTITY_API
        builder = lambda **kwargs: kwargs
        builder.round16_api_identity = adapter.ALLGATHER_BUILD_FUSED_INPUT_API
        with self.assertRaisesRegex(ValueError, "bfloat16"):
            adapter.AllGatherQ32LayerBinding(
                "layer",
                plan,
                owner,
                comm,
                provider,
                stream,
                builder,
                _source_token(preflight),
            )
        self.assertEqual(events, [])


class _Device:
    def __init__(self, value="npu:0"):
        self.value = value
        self.type = value.split(":", 1)[0]

    def __str__(self):
        return self.value


class _NonIterableTensor:
    """Tensor-shaped sentinel that detects accidental host materialization."""

    def __init__(self, name, shape, dtype, device="npu:0"):
        self.name = name
        self.shape = shape
        self.dtype = dtype
        self.device = _Device(device)
        self.iteration_attempts = 0

    def __iter__(self):
        self.iteration_attempts += 1
        raise AssertionError(f"{self.name} must not be iterated")


class AllGatherCommImpl(_TupleAllGatherCommImpl):
    def __init__(self, events, *, padded_rows=0, fail_at=None):
        super().__init__(events, fail_at=fail_at)
        self.padded_rows = padded_rows
        self.finalized = None

    def prepare(self, **kwargs):
        self.events.append("prepare")
        if self.fail_at == "prepare":
            raise RuntimeError("prepare failed")
        hidden = kwargs["hidden_states"]
        logits = kwargs["router_logits"]
        rows, width = hidden.shape
        prepared_rows = rows + self.padded_rows
        if self.padded_rows:
            hidden = _NonIterableTensor(
                "prepared-hidden", (prepared_rows, width), hidden.dtype,
                str(hidden.device),
            )
            logits = _NonIterableTensor(
                "prepared-router", (prepared_rows, 512), logits.dtype,
                str(logits.device),
            )
        return SimpleNamespace(
            hidden_states=hidden,
            router_logits=logits,
            mc2_mask=None,
            padded_hidden_states_shape=(prepared_rows, width)
            if self.padded_rows else None,
            pertoken_scale=None,
        )

    def fused_experts(self, *, fused_experts_input):
        self.events.append("fused_experts")
        if self.fail_at == "fused_experts":
            raise RuntimeError("fused failed")
        self.last_fused_input = fused_experts_input
        rows, width = fused_experts_input.hidden_states.shape
        output = _NonIterableTensor(
            "fused-output", (rows, width), "bfloat16",
            str(fused_experts_input.hidden_states.device),
        )
        return SimpleNamespace(routed_out=output)

    def finalize(self, **kwargs):
        self.events.append("finalize")
        if self.fail_at == "finalize":
            raise RuntimeError("finalize failed")
        hidden = kwargs["hidden_states"]
        rows, width = hidden.shape
        if kwargs["padded_hidden_states_shape"] is not None:
            rows -= self.padded_rows
        self.finalized = _NonIterableTensor(
            "finalized-output", (rows, width), hidden.dtype,
            str(hidden.device),
        )
        return self.finalized


class AllGatherQ32DeviceTensorAdapterTests(unittest.TestCase):
    def _fixture(self, *, padded_rows=0, fail_at=None, payload_override=None):
        adapter, preflight, loader, runner, _routing = _modules()
        events = []
        comm = AllGatherCommImpl(
            events, padded_rows=padded_rows, fail_at=fail_at
        )
        current = {"comm": comm, "stream": object()}
        plan = loader.plan_q32(tuple(range(32)))
        owner = SimpleNamespace(
            round16_selective_plan=plan,
            w13_weight=_Weight((32, 8, 16)),
            w2_weight=_Weight((32, 16, 8)),
            w13_bias=None,
            w2_bias=None,
            swiglu_limit=0.0,
            swiglu_alpha=1.0,
            swiglu_beta=0.0,
        )

        def current_comm():
            return current["comm"]

        current_comm.round16_api_identity = runner.SOURCE_BOUND_MOE_COMM_CONTEXT_API

        def current_stream():
            return current["stream"]

        current_stream.round16_api_identity = adapter.ALLGATHER_STREAM_IDENTITY_API

        def builder(**kwargs):
            events.append("build")
            if fail_at == "build":
                raise RuntimeError("build failed")
            return SimpleNamespace(**kwargs)

        builder.round16_api_identity = adapter.ALLGATHER_BUILD_FUSED_INPUT_API

        payload_calls = []

        def physical_payload(**kwargs):
            events.append("physical_payload")
            payload_calls.append(kwargs)
            if fail_at == "physical_payload":
                raise RuntimeError("physical payload failed")
            if payload_override is not None:
                return payload_override(kwargs)
            rows, top_k = kwargs["topk_ids"].shape
            device = str(kwargs["topk_ids"].device)
            return adapter.DevicePhysicalTopKPayload(
                topk_ids=_NonIterableTensor(
                    "physical-ids", (rows, top_k), "int32", device
                ),
                topk_weights=_NonIterableTensor(
                    "physical-weights", (rows, top_k), "bfloat16", device
                ),
            )

        physical_payload.round16_api_identity = (
            adapter.ALLGATHER_PHYSICAL_TENSOR_PAYLOAD_API
        )
        layer_binding = adapter.AllGatherQ32LayerBinding(
            layer_name="model.layers.0.mlp.experts",
            plan=plan,
            routed_owner=owner,
            comm_method=comm,
            comm_method_provider=current_comm,
            stream_identity_provider=current_stream,
            build_fused_experts_input=builder,
            source_preflight_token=_source_token(preflight),
        )
        tensor_binding = adapter.AllGatherQ32TensorBinding(
            layer=layer_binding,
            build_physical_topk_payload=physical_payload,
        )
        return (
            adapter,
            events,
            current,
            comm,
            payload_calls,
            adapter.AllGatherQ32DeviceTensorAdapter(tensor_binding),
        )

    @staticmethod
    def _inputs(rows=2, width=4):
        return (
            _NonIterableTensor("hidden", (rows, width), "bfloat16"),
            _NonIterableTensor("router", (rows, 512), "float32"),
            _NonIterableTensor("topk-ids", (rows, 10), "int32"),
            _NonIterableTensor("topk-weights", (rows, 10), "bfloat16"),
        )

    def test_live_tensor_path_preserves_device_output_without_iteration_or_float(self):
        adapter, events, _current, comm, calls, instance = self._fixture()
        hidden, router, topk_ids, topk_weights = self._inputs()
        prepared = instance.prepare(
            hidden, router, "model.layers.0.mlp.experts"
        )
        selected = adapter.DeviceSelectedTopK(
            topk_ids, topk_weights, prepared.valid_rows
        )
        result = instance.execute(
            prepared, selected, prepared.selector_hidden_states, ("work", 1)
        )
        self.assertIs(result.output, comm.finalized)
        self.assertTrue(result.route_weights_applied)
        self.assertFalse(result.routed_scaling_applied)
        self.assertEqual(
            events,
            ["prepare", "physical_payload", "build", "fused_experts", "finalize"],
        )
        self.assertEqual(calls[0]["resident_logical_ids"], tuple(range(32)))
        self.assertEqual(calls[0]["logical_to_physical"][:32], tuple(range(32)))
        self.assertEqual(calls[0]["logical_to_physical"][32:], (-1,) * 480)
        self.assertEqual(calls[0]["safe_physical_id"], 0)
        self.assertEqual(calls[0]["valid_rows"], (True, True))
        for tensor in (hidden, router, topk_ids, topk_weights, result.output):
            self.assertEqual(tensor.iteration_attempts, 0)
        self.assertFalse(instance.live_eligible)

    def test_optional_owner_swiglu_scalars_use_frozen_schema_defaults(self):
        adapter, _events, _current, comm, _calls, instance = self._fixture()
        owner = instance.layer.routed_owner
        owner.swiglu_limit = None
        owner.swiglu_alpha = None
        owner.swiglu_beta = None
        hidden, router, topk_ids, topk_weights = self._inputs()
        prepared = instance.prepare(
            hidden, router, "model.layers.0.mlp.experts"
        )
        selected = adapter.DeviceSelectedTopK(
            topk_ids, topk_weights, prepared.valid_rows
        )
        instance.execute(
            prepared, selected, prepared.selector_hidden_states, ("swiglu", 1)
        )
        self.assertEqual(comm.last_fused_input.swiglu_limit, 0.0)
        self.assertEqual(comm.last_fused_input.swiglu_alpha, 1.0)
        self.assertEqual(comm.last_fused_input.swiglu_beta, 0.0)

        owner.swiglu_limit = float("nan")
        prepared = instance.prepare(
            hidden, router, "model.layers.0.mlp.experts"
        )
        with self.assertRaisesRegex(ValueError, "swiglu_limit must be finite"):
            instance.execute(
                prepared,
                selected,
                prepared.selector_hidden_states,
                ("swiglu", 2),
            )

    def test_padding_metadata_reaches_device_payload_and_finalize_crops_output(self):
        adapter, events, _current, comm, calls, instance = self._fixture(
            padded_rows=1
        )
        hidden, router, _ids, _weights = self._inputs(rows=1)
        prepared = instance.prepare(
            hidden, router, "model.layers.0.mlp.experts"
        )
        self.assertEqual(prepared.valid_rows, (True, False))
        selected = adapter.DeviceSelectedTopK(
            _NonIterableTensor("padded-ids", (2, 10), "int32"),
            _NonIterableTensor("padded-weights", (2, 10), "bfloat16"),
            (True, False),
        )
        result = instance.execute(
            prepared, selected, prepared.selector_hidden_states, ("pad", 1)
        )
        self.assertEqual(calls[0]["valid_rows"], (True, False))
        self.assertEqual(result.output.shape, (1, 4))
        self.assertIs(result.output, comm.finalized)
        self.assertEqual(events[-1], "finalize")

    def test_tensor_dtype_device_shape_and_payload_markers_fail_closed(self):
        bad_inputs = (
            ("hidden dtype", _NonIterableTensor("hidden", (2, 4), "float16"), _NonIterableTensor("router", (2, 512), "float32"), "bfloat16"),
            ("hidden device", _NonIterableTensor("hidden", (2, 4), "bfloat16", "cpu"), _NonIterableTensor("router", (2, 512), "float32", "cpu"), "bfloat16"),
            ("router width", _NonIterableTensor("hidden", (2, 4), "bfloat16"), _NonIterableTensor("router", (2, 511), "float32"), "bfloat16"),
        )
        for label, hidden, router, _unused in bad_inputs:
            with self.subTest(label=label):
                _adapter, events, _current, _comm, _calls, instance = self._fixture()
                with self.assertRaises((TypeError, ValueError)):
                    instance.prepare(hidden, router, "model.layers.0.mlp.experts")
                self.assertEqual(events, [])

        def bad_payload(kwargs):
            rows, top_k = kwargs["topk_ids"].shape
            return adapter.DevicePhysicalTopKPayload(
                _NonIterableTensor("ids", (rows, top_k), "int64"),
                _NonIterableTensor("weights", (rows, top_k), "bfloat16"),
            )

        adapter, _events, _current, _comm, _calls, instance = self._fixture(
            payload_override=bad_payload
        )
        hidden, router, ids, weights = self._inputs()
        prepared = instance.prepare(hidden, router, "model.layers.0.mlp.experts")
        with self.assertRaisesRegex(ValueError, "int32"):
            instance.execute(
                prepared,
                adapter.DeviceSelectedTopK(ids, weights, prepared.valid_rows),
                prepared.selector_hidden_states,
                ("bad", 1),
            )

        adapter, _events, _current, _comm, _calls, instance = self._fixture()
        hidden, router, ids, _weights = self._inputs()
        prepared = instance.prepare(hidden, router, "model.layers.0.mlp.experts")
        with self.assertRaisesRegex(ValueError, "float32"):
            instance.execute(
                prepared,
                adapter.DeviceSelectedTopK(
                    ids,
                    _NonIterableTensor("weights", (2, 10), "float16"),
                    prepared.valid_rows,
                ),
                prepared.selector_hidden_states,
                ("bad", 2),
            )

    def test_tensor_failures_retire_one_shot_state(self):
        for fail_at in ("physical_payload", "build", "fused_experts", "finalize"):
            with self.subTest(fail_at=fail_at):
                adapter, _events, _current, comm, _calls, instance = self._fixture(
                    fail_at=fail_at
                )
                hidden, router, ids, weights = self._inputs()
                prepared = instance.prepare(
                    hidden, router, "model.layers.0.mlp.experts"
                )
                with self.assertRaisesRegex(RuntimeError, "failed"):
                    instance.execute(
                        prepared,
                        adapter.DeviceSelectedTopK(
                            ids, weights, prepared.valid_rows
                        ),
                        prepared.selector_hidden_states,
                        ("failure", fail_at),
                    )
                comm.fail_at = None
                next_hidden, next_router, _ids, _weights = self._inputs(1)
                next_prepared = instance.prepare(
                    next_hidden,
                    next_router,
                    "model.layers.0.mlp.experts",
                )
                instance.cleanup(next_prepared)

    def test_tensor_adapter_uses_distinct_device_api_identity(self):
        adapter, _events, _current, _comm, _calls, instance = self._fixture()
        self.assertEqual(
            instance.execute.round16_api_identity,
            adapter.ALLGATHER_DEVICE_WEIGHTED_PARTIAL_API,
        )
        self.assertNotEqual(
            instance.execute.round16_api_identity,
            _modules()[3].SOURCE_BOUND_NATIVE_WEIGHTED_PARTIAL_API,
        )


if __name__ == "__main__":
    unittest.main()
